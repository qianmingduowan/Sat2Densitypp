# python3.8
"""Contains the implementation of generator described in EG3D.

Paper: https://arxiv.org/pdf/2112.07945.pdf

Official PyTorch implementation: https://github.com/NVlabs/eg3d
"""

import torch
import torch.nn as nn

from models.utils.s2d_eg3d_model_helper import Generator_s2d as StyleGAN2Backbone
from models.utils.s2d_eg3d_model_helper import FullyConnectedLayer, ToRGBLayer
# from models.utils.eg3d_superres import SuperresolutionHybrid2X
from models.utils.eg3d_superres import Superresolution_s2d_Hybrid2X,Superresolution_s2d_Hybrid4X,Superresolution_s2d_Hybrid8XDC,Superresolution_s2d_sat
# from models.utils.eg3d_superres import SuperresolutionHybrid8XDC

# from models.rendering import PointSampler
from models.rendering import PointRepresenter
from models.rendering import PointIntegrator

from models.rendering.utils import sample_importance
from models.rendering.utils import unify_attributes
# import edict
from easydict import EasyDict as edict
from .sat2density_transform_eg3d import get_original_coord,Point_sampler_pano, Point_sampler_ortho, RGB_Reprerenter
from models.utils.eg3d_model_helper import MappingNetwork
from models.utils.s2d_eg3d_model_helper import SynthesisNetwork, Encoder
from einops import repeat, rearrange


class EG3DGenerator(nn.Module):

    def __init__(
        self,
        # Settings for mapping network.
        z_dim,
        label_dim,
        w_dim,
        mapping_layers,
        vanilla_nerf=False,
        no_sky_branch=False,
        sat_sr=False,
        infer_mode=False,
        z_dim_ill=512,
        z_and_w_dim_for_sat=False,
        c_dim_ill=270,
        image_channels=3,
        label_gen_conditioning_zero=False,
        label_scale=1.0,
        # Settings for triplane.
        triplane_resolution=256,
        triplane_channels=32 * 3,
        num_fp16_res=4,
        conv_clamp=256,
        coordinate_scale=1.0,
        # Settings for post CNN.
        sr_num_fp16_res=0,
        sr_channel_base=2**15,
        sr_channel_max=512,
        sr_antialias=True,
        sr_fused_modconv_default='inference_only',
        sr_noise_mode='none',
        # Settings for mlp network.
        mlp_hidden_dim=64,
        mlp_output_dim=32,
        mlp_lr_mul=1,
        # Settings for point sampling.
        point_sampling_kwargs=None,
        # Settings for ray marching.
        ray_marching_kwargs=None,
        # Settings for rendering.
        resolution=512,  # Image resolution of final output image.
        rendering_resolution=64,  # Resolution of NeRF rendering.
        num_importance=48,  # Number of points for importance sampling.
        # Settings for rendering options of evaluation.
        avg_camera_radius=2.7,
        avg_camera_pivot=[0, 0, 0.2],
        # Whether to use raw triplane axes as the official code.
        use_raw_triplane_axes=False,
        extract_rgb_features = False,
        mapping_kwargs=edict(),
        ):

        super().__init__()
        self.z_dim = z_dim
        self.label_dim = label_dim
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.resolution = resolution
        self.extract_rgb_features = extract_rgb_features
        self.infer_mode = infer_mode
        self.z_and_w_dim_for_sat = z_and_w_dim_for_sat
        self.vanilla_nerf = vanilla_nerf
        self.no_sky_branch = no_sky_branch
        self.sat_sr = sat_sr
        # Set up the rendering related module.
        if point_sampling_kwargs is None:
            point_sampling_kwargs = {}
        if ray_marching_kwargs is None:
            ray_marching_kwargs = {}
        self.point_integrator = PointIntegrator(**ray_marching_kwargs)

        if use_raw_triplane_axes:
            # The axes of the triplane in this case are actually incorrect
            # as stated in https://github.com/NVlabs/eg3d/issues/67
            triplane_axes = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                          [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                          [[0, 0, 1], [1, 0, 0], [0, 1, 0]]],
                                         dtype=torch.float32)
        else:
            triplane_axes = None

        self.point_representer = PointRepresenter(
            representation_type='triplane',
            triplane_axes=triplane_axes,
            coordinate_scale=coordinate_scale)

        # Set up the reference representation generator.
        self.backbone = StyleGAN2Backbone(
            z_dim,
            label_dim,
            w_dim,
            img_resolution=triplane_resolution,
            img_channels=triplane_channels,
            num_fp16_res=num_fp16_res,
            conv_clamp=conv_clamp,
            max_block=4,
            mapping_kwargs=dict(num_layers=mapping_layers,mapping_sat=False))
        # self.histo_mapping = MappingNetwork(z_dim=0,
        #                                 c_dim=270,
        #                                 w_dim=512,
        #                                 num_ws=1,
        #                                 num_layers=3,
        #                                 )
        if not self.no_sky_branch:
            self.sky_synthesis = SynthesisNetwork(
                w_dim=w_dim,
                img_resolution=128,
                img_channels=triplane_channels//3,
                num_fp16_res=num_fp16_res,
                conv_clamp=conv_clamp,
                for_pano=True,
            )
            num_ws_sky = self.sky_synthesis.num_ws
        else:
            num_ws_sky = 8
        assert z_dim_ill in [0, 512]
        assert c_dim_ill in [0, 270, 272]
        self.z_dim_ill = z_dim_ill
        self.c_dim_ill = c_dim_ill
        self.sky_img_mapping = MappingNetwork(z_dim=z_dim_ill,
                                            c_dim=c_dim_ill,
                                            w_dim=512,
                                            num_ws=num_ws_sky,
                                            num_layers=3,
                                            )

        # Set up the post CNN.
        self.sr_noise_mode = sr_noise_mode
        sr_kwargs = {
            'channels': mlp_output_dim,
            'img_resolution': resolution,
            'sr_num_fp16_res': sr_num_fp16_res,
            'sr_antialias': sr_antialias,
            'channel_base': sr_channel_base,
            'channel_max': sr_channel_max,
            'fused_modconv_default': sr_fused_modconv_default
        }
        if resolution == 128:
            self.post_cnn = Superresolution_s2d_Hybrid2X(**sr_kwargs)
        elif resolution == 256: # avg of synthesized h,w (128,512)
            self.post_cnn = Superresolution_s2d_Hybrid4X(**sr_kwargs)
        elif resolution == 512:
            self.post_cnn = Superresolution_s2d_Hybrid8XDC(**sr_kwargs)
        else:
            raise TypeError(f'Unsupported image resolution: {resolution}!')
        print('resolution',resolution)
        if self.sat_sr:
            self.post_cnn_sat = Superresolution_s2d_sat(**sr_kwargs)

        # Set up the MLP network.
        mlp_input_dim = triplane_channels // 3
        if self.extract_rgb_features:
            mlp_input_dim += 3

        # self.mlp = MLPNetwork(input_dim=mlp_input_dim,
        #                       hidden_dim=mlp_hidden_dim,
        #                       output_dim=mlp_output_dim,
        #                       lr_mul=mlp_lr_mul)
        if not self.vanilla_nerf:
            self.mlp = MLPNetwork2(input_dim=mlp_input_dim,
                                    hidden_dim=mlp_hidden_dim,
                                    output_dim=mlp_output_dim,
                                    lr_mul=mlp_lr_mul,
                                    z_dim=z_dim,)
        else:
            self.mlp = MLPNetwork_vanilla(input_dim=mlp_input_dim,
                                    hidden_dim=mlp_hidden_dim,
                                    output_dim=mlp_output_dim,
                                    lr_mul=mlp_lr_mul,
                                    z_dim=z_dim,)

        # Set up some other arguments.
        self.register_buffer('rendering_resolution',
                             torch.tensor(rendering_resolution))
        self.num_importance = num_importance
        self.eval_rendering_options = {
            'avg_camera_radius': avg_camera_radius,
            'avg_camera_pivot': avg_camera_pivot
        }
        if point_sampling_kwargs.data_type.lower() == 'cvusa':
            full=False
        else:
            full=True
        self.pano_direction = torch.from_numpy(get_original_coord(W=512//2,H=128//2,full=full)).unsqueeze(0).cuda().float()
        self.point_sampler = Point_sampler_pano(pano_direction=self.pano_direction,**point_sampling_kwargs)
        self.point_sampler_sat = Point_sampler_ortho(num_points=point_sampling_kwargs.num_points)
        if self.extract_rgb_features: # not used any more, can be deleted
            self.rgb_reprerenter  = RGB_Reprerenter()
        self.encoder = Encoder(img_resolution=256, img_channels=3,max_block=4, model_kwargs={'output_mode': None})




    def mapping(self,
                z,
                label,
                sat_img,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False):
            # label = torch.zeros_like(label)
        results = self.backbone.mapping(z,
                                        label,
                                        sat_img,
                                        truncation_psi=truncation_psi,
                                        truncation_cutoff=truncation_cutoff,
                                        update_emas=update_emas)
        return results
    
    def synthesis_tri_plane(self,
                    wp,
                    sat_img=None,
                    update_emas=False,
                    **synthesis_kwargs):
        if 'origin_H_W' in synthesis_kwargs.keys():
            synthesis_kwargs.pop('origin_H_W')
        feature =  self.encoder(sat_img)['feature']
        triplanes = self.backbone.synthesis(wp,
                                            x = feature,
                                            update_emas=update_emas,
                                            **synthesis_kwargs)
        triplanes = triplanes.view(triplanes.shape[0], 3, -1,
                                   triplanes.shape[-2], triplanes.shape[-1])
        return triplanes
    
    def from_point_sampling2result(self,
                                        point_sampling_result,
                                        triplanes,
                                        sat_img,
                                        w_sky=None,
                                        update_emas=False,
                                          ):
        points = point_sampling_result['points_world']  # [N, H, W, K, 3]
        ray_dirs = point_sampling_result['rays_world']  # [N, H, W, 3]
        radii_coarse = point_sampling_result['radii']  # [N, H, W, K]
        ray_origins = point_sampling_result['ray_origins'] # [N, 3]

        _, H, W, K, _ = points.shape
        R = H * W
        points_coarse = rearrange(points, 'n h w k c -> n (h w) k c')  # [N, R, K, 3]
        points = rearrange(points, 'n h w k c -> n (h w k) c')  # [N, R * K, 3]
        ray_dirs = rearrange(ray_dirs, 'n h w c -> n (h w) c')
        if len(ray_origins.shape) == 4:
            ray_origins = rearrange(ray_origins, 'n h w c -> n (h w) c')
        elif len(ray_origins.shape) == 2:
            ray_origins = repeat(ray_origins, 'n c -> n (h w) c', h=R, w=1)
        radii_coarse = rearrange(radii_coarse, 'n h w k -> n (h w) k 1')

        point_features = self.point_representer(
            points, ref_representation=triplanes)  # [N, R * K, C]
        if self.extract_rgb_features:
            rgb_features = self.rgb_reprerenter(points, sat_img)
            point_features = torch.cat([point_features, rgb_features], dim=-1)
        color_density_result = self.mlp(point_features,w_sky)

        densities_coarse = color_density_result['density']  # [N, R * K, 1]
        colors_coarse = color_density_result['color']  # [N, R * K, C]
        densities_coarse = rearrange(densities_coarse, 'n (r k) c -> n r k c', r=R, k=K)
        colors_coarse = rearrange(colors_coarse, 'n (r k) c -> n r k c', r=R, k=K)

        if self.num_importance > 0:
            # Do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                     densities_coarse,
                                                     radii_coarse)
            weights = rendering_result['weight']

            # Importance sampling.
            radii_fine = sample_importance(radii_coarse,
                                           weights,
                                           self.num_importance,
                                           smooth_weights=True)
            points = ray_origins.unsqueeze(
                -2) + radii_fine * ray_dirs.unsqueeze(
                -2)  # [N, R, num_importance, 3]
            points_fine = points
            points = rearrange(points, 'n r k c -> n (r k) c')  # [N, R * num_importance, 3]

            point_features = self.point_representer(
                points, ref_representation=triplanes)
            if self.extract_rgb_features:
                rgb_features = self.rgb_reprerenter(points, sat_img)
                point_features = torch.cat([point_features, rgb_features], dim=-1)
            color_density_result = self.mlp(point_features,w_sky)

            densities_fine = color_density_result['density']
            colors_fine = color_density_result['color']
            densities_fine = rearrange(densities_fine, 'n (r k) c -> n r k c', r=R, k=self.num_importance)
            colors_fine = rearrange(colors_fine, 'n (r k) c -> n r k c', r=R, k=self.num_importance)

            # Gather coarse and fine results together.
            (all_radiis, all_colors, all_densities,
             all_points) = unify_attributes(radii_coarse,
                                            colors_coarse,
                                            densities_coarse,
                                            radii_fine,
                                            colors_fine,
                                            densities_fine,
                                            points1=points_coarse,
                                            points2=points_fine)

            # Do the integration along the fine pass.
            rendering_result = self.point_integrator(all_colors,
                                                     all_densities,
                                                     all_radiis)

        else:
            # Only do the integration along the coarse pass.
            rendering_result = self.point_integrator(colors_coarse,
                                                     densities_coarse,
                                                     radii_coarse)
            all_points = points_coarse  # [N, R, K, 3]

        feature_samples = rendering_result['composite_color']
        radii_samples = rendering_result['composite_radial_dist']

        # Reshape to keep consistent with 'raw' NeRF-rendered image.
        feature_image = rearrange(feature_samples, 'n (h w) c -> n c h w', h=H, w=W).contiguous()  # [N, C, H, W]
        # feature_image = feature_samples.permute(0, 2, 1).reshape(
        #     N, feature_samples.shape[-1], H, W).contiguous()
        image_radii = rearrange(radii_samples, 'n (h w) c -> n c h w', h=H, w=W).contiguous()  # [N, 1, H, W]
        # image_radii = radii_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # # Get image depth.
        # # import ipdb; ipdb.set_trace()
        # all_points_camera = all_points - ray_origins.unsqueeze(-2)
        # points_camera_z = all_points_camera[..., -1:].abs()  # [N, R, K, 1]
        # points_camera_z = (points_camera_z[:, :, :-1, :] +
        #                    points_camera_z[:, :, 1:, :]) / 2  # [N, R, K-1, 1]
        # point_weights = rendering_result['weight']  # [N, R, K-1, 1]
        # # point_alpha = point_weights.sum(dim=-2)  # [N, R, 1]
        image_alpha = rearrange(rendering_result['opacity'], 'n (h w) c -> n c h w', h=H, w=W).contiguous()
        # image_alpha = rendering_result['opacity'].permute(0, 2, 1).reshape(N, 1, H, W)
        # image_alpha = point_alpha
        # image_depth = torch.sum(point_weights * points_camera_z,
        #                         dim=-2)  # [N, R, 1]
        # image_depth = rearrange(image_depth, 'n (h w) c -> n c h w', h=H, w=W).contiguous()
        # rendering_result[composite_radial_dist] is image_depth
        image_depth = rearrange(rendering_result['composite_radial_dist'], 'n (h w) c -> n c h w', h=H, w=W).contiguous()

        # Run the post CNN to get final image.
        # Here, the post CNN is a super-resolution network.
        rgb_image = feature_image[:, :3]
        result = edict()
        result.feature_raw = feature_image
        result.alpha_raw = image_alpha
        result.image_raw = rgb_image
        result.image_depth = image_depth
        result.image_radii = image_radii
        if 'idx' in point_sampling_result.keys():
            result.idx = point_sampling_result['idx']
        return result


    def from_triplane_2_sat(self,
                            triplanes,
                            sat_img=None,
                            update_emas=False,
                            random_crop=True,
                            w=None,
                            ):
        '''
        sat_img: only used when extract_rgb_features is True
        '''
        N = triplanes.shape[0]
        point_sampling_result = self.point_sampler_sat(batch_size=N,random_crop=random_crop)

        results = self.from_point_sampling2result(point_sampling_result,
                                                        triplanes,
                                                        sat_img,
                                                        update_emas=update_emas,
                                                        w_sky=w,
                                                        )
        if self.sat_sr:
            sr_kwargs = {
            
                }
            if w is None:
                # w all zero b , 1 , 512
                w = torch.zeros((N, 1, self.w_dim), device=triplanes.device, dtype=triplanes.dtype)
            sr_image = self.post_cnn_sat(results.feature_raw[:,:3].clone(),
                    results.feature_raw.clone(),
                    w,    # changed for sat2density
                    noise_mode=self.sr_noise_mode,
                    **sr_kwargs
                    )
            # resize back to //2
            h_size = sr_image.shape[-2]
            w_size = sr_image.shape[-1]
            sr_image = torch.nn.functional.interpolate(sr_image, size=(h_size//2, w_size//2), mode='bilinear', align_corners=False)
            results.feature_raw = sr_image
            results.image_raw = sr_image[:,:3].clone()
        return results


    def from_triplane_2_grd(self,
                            wp,
                            triplanes,
                            sat_img=None,
                            origin_H_W = None,
                            w_sky = None,
                            update_emas=False,
                            **synthesis_kwargs
                            ):
        N = triplanes.shape[0]
        point_sampling_result = self.point_sampler(batch_size=N,origin_H_W=origin_H_W)
        results = self.from_point_sampling2result(point_sampling_result,
                                                        triplanes,
                                                        sat_img,
                                                        w_sky=w_sky,
                                                        update_emas=update_emas,
                                                        )

        sr_kwargs = {
            k: synthesis_kwargs[k]
            for k in synthesis_kwargs.keys() if k != 'noise_mode'
        }
        if not self.no_sky_branch:
            sky_img_feature =  self.sky_synthesis(w_sky,     ## changed for simplify                                         
                                    update_emas=update_emas,
                                    **synthesis_kwargs)
            h_size = sky_img_feature.shape[-2]
            # sky_img_feature = torch.clamp(sky_img_feature, -1, 1)
            results.sky_img = sky_img_feature[:,:3].clone()
            foreground_feature = results.feature_raw
            results.image_raw = foreground_feature[:,:3].clone()
            # [-1,1] -> [0,2] -> [6,8]
            sky_img_feature = sky_img_feature+7
            alpha_raw = results.alpha_raw
            if  self.infer_mode and  not self.training:
                alpha_raw[:,:,h_size//2:,:] = 1
                alpha_raw[alpha_raw < 0.1] = 0
            rgb_feature_compo = (foreground_feature +7) * alpha_raw + (sky_img_feature) * (1 - alpha_raw)  -7 # alpha compositing in data range [-1,1]
            rgb_img_compo = rgb_feature_compo[:,:3]
        else:
            rgb_feature_compo = results.feature_raw
            rgb_img_compo = results.feature_raw[:,:3]
        results.image_raw_compo = rgb_img_compo
        rgb_img_compo_copy  = rgb_img_compo.clone()
        rgb_feature_compo_copy = rgb_feature_compo.clone()
        sr_image = self.post_cnn(rgb_img_compo_copy,
                                 rgb_feature_compo_copy,
                                 wp,    # changed for sat2density
                                 noise_mode=self.sr_noise_mode,
                                 **sr_kwargs)
        results.image = sr_image

        return results


    def forward(self,
                z,
                label,
                sat_img,
                histo_sky,
                label_swapped=None,
                style_mixing_prob=0,
                truncation_psi=1,
                truncation_cutoff=None,
                update_emas=False,
                density_reg=False,
                sat_syn=False,
                grd_syn=True,
                coordinates=None,
                origin_H_W=None,
                random_crop=True,
                **synthesis_kwargs):
        wp = self.mapping(z,
                          label if label_swapped is None else label_swapped,
                          sat_img,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        # w_sty = self.histo_mapping(None,
        #                         histo_sky,
        #                         truncation_psi=1,
        #                         truncation_cutoff=None,
        #                         update_emas=False)


        # if style_mixing_prob > 0:
        #     cutoff = torch.empty([], dtype=torch.int64,
        #                          device=wp.device).random_(1, wp.shape[1])
        #     cutoff = torch.where(
        #         torch.rand([], device=wp.device) < style_mixing_prob, cutoff,
        #         torch.full_like(cutoff, wp.shape[1]))
        #     wp[:, cutoff:] = self.mapping(torch.randn_like(z),
        #                                   label,
        #                                   update_emas=update_emas)[:, cutoff:]
        if density_reg:
            # Only for density regularization in training process.
            assert coordinates is not None
            sample_density = self.sample_mixed(coordinates,
                                               wp,
                                               sat_img = sat_img,
                                               update_emas=False)['density']

            results = {'wp': wp, 'sample_density': sample_density}
        else:
            results = edict({'wp': wp })
            triplanes = self.synthesis_tri_plane(wp,
                                            sat_img=sat_img,
                                            update_emas=update_emas,
                                            **synthesis_kwargs)
            results.triplanes = triplanes
            if grd_syn:
                w_sky = self.sky_img_mapping(z,
                                    histo_sky,
                            truncation_psi=1,
                            truncation_cutoff=None,
                            update_emas=False)

            if sat_syn:
                if self.z_and_w_dim_for_sat:
                    gen_output = self.from_triplane_2_sat(triplanes,
                                                    sat_img=sat_img,
                                                    update_emas=update_emas,
                                                    random_crop=random_crop,
                                                    w=w_sky,
                                                    **synthesis_kwargs)
                else:
                    gen_output = self.from_triplane_2_sat(triplanes,
                                                        sat_img=sat_img,
                                                        update_emas=update_emas,
                                                        random_crop=random_crop,
                                                        **synthesis_kwargs)
                results.sat_img_raw = gen_output.image_raw
                results.sat_dep = gen_output.image_depth
                # norm_sat_dep_relative = ((gen_output.image_depth - gen_output.image_depth.min()) / (gen_output.image_depth.max() - gen_output.image_depth.min())) *2 -1
                # It is just a continuation of the name of the previous version, but actually has no depth.
                results.cat_sat_img_dep = results.sat_img_raw
                # torch.cat([ results.sat_img_raw, norm_sat_dep_relative], dim=1)
                if 'idx' in gen_output.keys():
                    results.idx = gen_output.idx
            if grd_syn:
                gen_output = self.from_triplane_2_grd(wp,
                                                    triplanes,
                                                    sat_img=sat_img,
                                                    w_sky=w_sky,
                                                    update_emas=update_emas,
                                                    origin_H_W=origin_H_W,
                                                    **synthesis_kwargs)
                results.update(gen_output)

        return results

    def sample(self,
               coordinates,
               z,
               sat_img,
               truncation_psi=1,
               truncation_cutoff=None,
               update_emas=False,
               **synthesis_kwargs):
        """Samples densities and colors.

        By feeding the coordinates and latent codes into the trained network,
        one can sample densities and colors. This function is primarily used
        for evaluation purposes.
        """
        wp = self.mapping(z,
                          None,
                          sat_img,
                          truncation_psi=truncation_psi,
                          truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        feature =  self.encoder(sat_img)['feature']
        triplanes = self.backbone.synthesis(wp,
                                            x = feature,
                                            update_emas=update_emas,
                                            **synthesis_kwargs)
        triplanes = triplanes.view(triplanes.shape[0], 3, -1,
                                   triplanes.shape[-2], triplanes.shape[-1])

        point_features = self.point_representer(
            coordinates, ref_representation=triplanes)
        if self.extract_rgb_features:
            rgb_features = self.rgb_reprerenter(coordinates, sat_img)
            point_features = torch.cat([point_features, rgb_features], dim=-1)
        color_density_result = self.mlp(point_features)

        return color_density_result

    def sample_mixed(self,
                     coordinates,
                     wp,
                     truncation_psi=1,
                     truncation_cutoff=None,
                     sat_img=None,
                     update_emas=False,
                     **synthesis_kwargs):
        feature =  self.encoder(sat_img)['feature']
        triplanes = self.backbone.synthesis(wp,
                                            x = feature,
                                            update_emas=update_emas,
                                            **synthesis_kwargs)
        triplanes = triplanes.view(triplanes.shape[0], 3, -1,
                                   triplanes.shape[-2], triplanes.shape[-1])

        point_features = self.point_representer(
            coordinates, ref_representation=triplanes)
        if self.extract_rgb_features:
            rgb_features = self.rgb_reprerenter(coordinates, sat_img)
            point_features = torch.cat([point_features, rgb_features], dim=-1)
        color_density_result = self.mlp(point_features)

        return color_density_result


class MLPNetwork(nn.Module):
    """Defines fully-connected layer head in EG3D."""

    def __init__(self, input_dim, hidden_dim, output_dim, lr_mul):
        super().__init__()

        self.net = nn.Sequential(
            FullyConnectedLayer(input_dim, hidden_dim, lr_multiplier=lr_mul),
            nn.Softplus(),
            FullyConnectedLayer(hidden_dim,
                                1 + output_dim,
                                lr_multiplier=lr_mul))

    def forward(self, point_features):
        N, M, C = point_features.shape
        point_features = point_features.view(N * M, C)

        y = self.net(point_features)
        y = y.view(N, M, -1)

        color = torch.sigmoid(y[..., 1:]) * (1 + 2 * 0.001) - 0.001
        density = y[..., 0:1]

        return {'color': color, 'density': density}
    
class MLPNetwork2(nn.Module):
    """Defines fully-connected layer head in EG3D."""

    def __init__(self, input_dim, hidden_dim, output_dim, lr_mul, z_dim=512):
        super().__init__()

        self.net0 = FullyConnectedLayer(input_dim, hidden_dim, lr_multiplier=lr_mul)
        self.net0_act = nn.Softplus()
        self.net1_feature    = FullyConnectedLayer(hidden_dim, output_dim,lr_multiplier=lr_mul)
        self.net1_density    = FullyConnectedLayer(hidden_dim, 1,lr_multiplier=lr_mul)
        # self.gamma = FullyConnectedLayer(z_dim, output_dim, lr_multiplier=lr_mul)
        # self.beta  = FullyConnectedLayer(z_dim, output_dim, lr_multiplier=lr_mul)
        self.grd_color_convert = ToRGBLayer(in_channels=output_dim,
                                                  out_channels=output_dim,
                                                  w_dim=512,
                                                  )

    def forward(self, point_features, style=None):
        

        N, M, C = point_features.shape
        point_features = point_features.view(N * M, C)
        y = self.net0(point_features)
        y = self.net0_act(y)
        color = self.net1_feature(y).view(N, M, -1)
        if style is not None:
            style = style[:,-1]
        else:
            # send to point_features.device
            style = torch.ones([N, 512]).to(point_features.device)
        color = rearrange(color, 'n M c -> n c M 1')
        color = self.grd_color_convert(color,style)
        color = rearrange(color, 'n c M 1 -> n M c')
        density = self.net1_density(y).view(N, M, -1)
        color = torch.sigmoid(color) * (1 + 2 * 0.001) - 0.001
        return {'color': color, 'density': density}
    

class MLPNetwork_vanilla(nn.Module):
    """Defines fully-connected layer head in EG3D."""

    def __init__(self, input_dim, hidden_dim, output_dim, lr_mul, z_dim=512):
        super().__init__()

        self.net0 = FullyConnectedLayer(input_dim, hidden_dim, lr_multiplier=lr_mul)
        self.net0_act = nn.Softplus()
        self.net1_feature    = FullyConnectedLayer(hidden_dim, output_dim,lr_multiplier=lr_mul)
        self.net1_density    = FullyConnectedLayer(hidden_dim, 1,lr_multiplier=lr_mul)
        # self.gamma = FullyConnectedLayer(z_dim, output_dim, lr_multiplier=lr_mul)
        # self.beta  = FullyConnectedLayer(z_dim, output_dim, lr_multiplier=lr_mul)
        self.grd_color_convert = FullyConnectedLayer(output_dim,output_dim,lr_multiplier=lr_mul)
        print('Using MLPNetwork_vanilla, which is not used in the original s2dpp code!')

    def forward(self, point_features, style=None):
        

        N, M, C = point_features.shape
        point_features = point_features.view(N * M, C)
        y = self.net0(point_features)
        y = self.net0_act(y)
        color = self.net1_feature(y)
        # if style is not None:
        #     style = style[:,-1]
        # else:
        #     # send to point_features.device
        #     style = torch.ones([N, 512]).to(point_features.device)
        # color = rearrange(color, 'n M c -> n c M 1')
        color = self.grd_color_convert(color).view(N, M, -1)
        # color = rearrange(color, 'n c M 1 -> n M c')
        density = self.net1_density(y).view(N, M, -1)
        color = torch.sigmoid(color) * (1 + 2 * 0.001) - 0.001
        return {'color': color, 'density': density}

