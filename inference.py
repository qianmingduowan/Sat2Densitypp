# python3.8
"""Inference script for Sat2Density++ - Generate street-view videos from satellite imagery."""

import os
import argparse
import json
import numpy as np
import cv2
import tqdm

import torch
import csv
import torchvision
from easydict import EasyDict as edict

from utils.misc import bool_parser
from models import build_model
from utils.sky_utils import make_histo

from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
from PIL import ImageDraw
from PIL import Image

# Color list for trajectory visualization
colors = [
    (255, 0, 0),   # Red
    (255, 127, 0), # Orange
    (255, 255, 0), # Yellow
    (0, 255, 0),   # Green
    (0, 0, 255)    # Blue
]


def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):
    """Save 3D mesh as OBJ file."""
    try:
        import open3d as o3d
    except ImportError:
        print("Warning: open3d not installed. Saving as simple OBJ format.")
        # Simple OBJ export without open3d
        with open(fpath, 'w') as f:
            # Apply transformations
            pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            facenp_fx3 = facenp_fx3[:, [2, 1, 0]]
            
            if colornp_px3.dtype != np.float64:
                colornp_px3 = colornp_px3.astype(np.float32) / 255.0
            
            # Write vertices with colors
            for v, c in zip(pointnp_px3, colornp_px3):
                f.write(f'v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n')
            
            # Write faces (OBJ uses 1-based indexing)
            for face in facenp_fx3:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
        print(f"Mesh saved to {fpath}")
        return
    
    # Apply transformations
    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    # Create Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pointnp_px3)
    mesh.triangles = o3d.utility.Vector3iVector(facenp_fx3)

    if colornp_px3.dtype != np.float64:
        colornp_px3 = colornp_px3.astype(np.float32) / 255.0    
    mesh.vertex_colors = o3d.utility.Vector3dVector(colornp_px3)

    # Export to OBJ
    o3d.io.write_triangle_mesh(fpath, mesh, write_vertex_normals=False)
    print(f"Mesh saved to {fpath}")


def interpolate_color(factor, colors=colors):
    """Interpolate color based on factor for trajectory visualization."""
    num_colors = len(colors)
    segment = factor * (num_colors - 1)
    index = int(segment)
    local_factor = segment - index
    color_start = colors[index]
    color_end = colors[min(index + 1, num_colors - 1)]
    return tuple(int(start + (end - start) * local_factor) for start, end in zip(color_start, color_end))


def create_voxel(N=256, voxel_corner=[0, 0, 0], voxel_length=2.0):
    """Creates a voxel grid for 3D shape extraction."""
    voxel_origin = np.array(voxel_corner) - voxel_length / 2
    voxel_size = voxel_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    grid = torch.zeros(N ** 3, 3)

    # Get the x, y, z index of each point in the grid.
    grid[:, 2] = overall_index % N
    grid[:, 1] = (overall_index.float() / N) % N
    grid[:, 0] = ((overall_index.float() / N) / N) % N

    # Get the x, y, z coordinate of each point in the grid.
    grid[:, 0] = (grid[:, 0] * voxel_size) + voxel_origin[2]
    grid[:, 1] = (grid[:, 1] * voxel_size) + voxel_origin[1]
    grid[:, 2] = (grid[:, 2] * voxel_size) + voxel_origin[0]

    return {
        'voxel_grid': grid.unsqueeze(0),
        'voxel_origin': voxel_origin,
        'voxel_size': voxel_size
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sat2Density++ Inference.')

    # Input & Output paths
    parser.add_argument('--sat_img_path', type=str, required=True,
                        help='Path to the satellite image.')
    parser.add_argument('--sky_path', type=str, required=True,
                        help='Path to sky panorama image for illumination.')
    parser.add_argument('--position_path', default=None, type=str,
                        help='Path to the trajectory CSV file (e.g., pixels.csv).')
    parser.add_argument('--work_dir', type=str,
                        default='work_dirs/visualize_result',
                        help='Working directory to save results.')

    # Output options
    parser.add_argument('--save_shape', type=bool_parser, default=True,
                        help='Whether to extract and save 3D mesh.')
    parser.add_argument('--save_video', type=bool_parser, default=False,
                        help='Whether to generate and save video.')
    parser.add_argument('--save_sky', type=bool_parser, default=False,
                        help='Whether to save sky image.')
    parser.add_argument('--save_sat', type=bool_parser, default=False,
                        help='Whether to save satellite-view rendering.')
    parser.add_argument('--save_street', type=bool_parser, default=False,
                        help='Whether to save street-view rendering.')
    parser.add_argument('--visual_direction', type=bool_parser, default=False,
                        help='Whether to visualize viewing direction on street view.')

    # Model parameters
    parser.add_argument('--truncation_psi', type=float, default=0.7,
                        help='Truncation psi for style mixing.')
    parser.add_argument('--truncation_cutoff', type=int, default=14,
                        help='Truncation cutoff.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--infer_mode', type=bool_parser, default=False,
                        help='Whether to use inference mode.')

    # Rendering parameters
    parser.add_argument('--rendering_resolution', type=int, default=64,
                        help='Neural rendering resolution.')
    
    # Shape extraction parameters
    parser.add_argument('--shape_res', type=int, default=256,
                        help='Resolution of the shape cube grid.')

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_path = args.sat_img_path  

    # Load and preprocess satellite image
    sat_img = Image.open(img_path).convert('RGB').resize((256, 256), Image.BILINEAR)
    sat_img = np.array(sat_img).astype(np.float32)
    sat_img = torch.from_numpy(sat_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
    sat_img = sat_img / 255.0 * 2.0 - 1.0  # Normalize to [-1,1]

    # Determine dataset type and parameters
    if 'CVACT' in img_path or 'cvact' in img_path.lower():
        data_path = os.path.dirname(os.path.dirname(img_path))
        half_pixel_num = 128
        dataset_tag = "cvact"
    elif 'VIGOR' in img_path or 'vigor' in img_path.lower():
        data_path = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))
        half_pixel_num = 320
        dataset_tag = "vigor"
    elif 'CVUSA' in img_path or 'cvusa' in img_path.lower():
        data_path = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))
        half_pixel_num = 375
        dataset_tag = "cvusa"
    else:
        raise ValueError(f"Cannot determine dataset type from path: {img_path}")

    # Calculate camera position based on dataset type
    if 'vigor' in img_path.lower() or 'VIGOR' in img_path:
        # For VIGOR, use default center position
        # You can modify offset_h and offset_w for different positions
        city_resolution = edict()
        city_resolution.Chicago      = 0.111
        city_resolution.NewYork      = 0.113
        city_resolution.SanFrancisco = 0.118
        city_resolution.Seattle      = 0.115
        
        # Assume center position by default
        city = 'Chicago'  # Default city
        if 'Chicago' in img_path:
            city = 'Chicago'
        elif 'NewYork' in img_path:
            city = 'NewYork'
        elif 'SanFrancisco' in img_path:
            city = 'SanFrancisco'
        elif 'Seattle' in img_path:
            city = 'Seattle'
        
        # Not very important; the difference is small compared to using a fixed position z.
        resolution = city_resolution[city]
        real_world_resolution = resolution * half_pixel_num
        position_z = 3./ real_world_resolution * 2 -1
        # Default to center (0, 0)
        position = np.array([0, 0, position_z])
        
    elif 'cvact' in img_path.lower() or 'CVACT' in img_path:
        half_real_world_resolution = 15
        position_z = (3./ half_real_world_resolution) -1
        position = np.array([0, 0, position_z])
        
    elif 'cvusa' in img_path.lower() or 'CVUSA' in img_path:
        position_z = -0.8
        position = np.array([0, 0, -0.8])
    else:
        raise ValueError('Dataset type not recognized from path')
    
    position = torch.from_numpy(position).unsqueeze(0).float().cuda()

    # Load sky image histogram
    if args.save_sky or args.save_street or args.save_video:
        histo_sky, masked_sky_img = make_histo(args.sky_path, True, True)
        if histo_sky.ndim == 1:
            histo_sky = histo_sky.unsqueeze(0)
        histo_sky = histo_sky.to(device)
    else:
        masked_sky_img = None

    # Load trajectory for video generation
    position_vid = []
    
    # Determine trajectory path
    if args.position_path and os.path.exists(args.position_path):
        # If position_path is provided and exists, use it directly
        position_path = args.position_path
    elif args.position_path:
        # If position_path is relative, join with work_dir
        position_path = os.path.join(args.work_dir, os.path.basename(img_path).rsplit('.', 1)[0], args.position_path)
    else:
        # Default: look for pixels.csv in work_dir
        position_path = os.path.join(args.work_dir, os.path.basename(img_path).rsplit('.', 1)[0], 'pixels.csv')
    
    work_dir = os.path.join(
        args.work_dir,
        os.path.basename(img_path).rsplit('.', 1)[0],
        f"{dataset_tag}seed{args.seed}",
    )
    
    os.makedirs(work_dir, exist_ok=True)
    if args.save_video:
        assert os.path.exists(position_path), f"{position_path} not exists"
        position_z_val = position[0, 2].item()
        
        with open(position_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x = float(row['h'])
                y = float(row['w'])
                position_vid.append(((x-half_pixel_num)/half_pixel_num, (y-half_pixel_num)/half_pixel_num, position_z_val))

    os.makedirs(args.work_dir, exist_ok=True)

    # Load model (config + weights from dataset checkpoints)
    print("Loading model from dataset checkpoints...")
    if 'cvact' in img_path.lower():
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "cvact")
    elif 'cvusa' in img_path.lower():
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "cvusa")
    elif 'vigor' in img_path.lower():
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "vigor")
    else:
        raise ValueError(f"Cannot determine dataset type from path: {img_path}")

    cfg_path = os.path.join(ckpt_dir, "generator_config.json")
    weights_path = os.path.join(ckpt_dir, "generator_smooth.pth")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    with open(cfg_path, "r") as f:
        gen_cfg = json.load(f)
    if isinstance(gen_cfg.get("point_sampling_kwargs"), dict):
        gen_cfg["point_sampling_kwargs"] = edict(gen_cfg["point_sampling_kwargs"])
    if isinstance(gen_cfg.get("ray_marching_kwargs"), dict):
        gen_cfg["ray_marching_kwargs"] = edict(gen_cfg["ray_marching_kwargs"])
    if isinstance(gen_cfg.get("mapping_kwargs"), dict):
        gen_cfg["mapping_kwargs"] = edict(gen_cfg["mapping_kwargs"])

    state_dict = {"model_kwargs_init": {"generator": gen_cfg}, "models": {}}
    state_dict["models"]["generator_smooth"] = torch.load(weights_path, map_location=device)
    G = build_model(**state_dict['model_kwargs_init']['generator'])
    G.load_state_dict(state_dict['models']['generator_smooth'], strict=False)
    G.eval().cuda()
    G_kwargs = dict(truncation_psi=args.truncation_psi,
                    truncation_cutoff=args.truncation_cutoff,
                    noise_mode='random')
    G.infer_mode = args.infer_mode
    assert (state_dict['models']['generator_smooth']['rendering_resolution'] ==
            args.rendering_resolution)

    max_street_depth = 0
    results = edict()
    
    with torch.no_grad():
        # Generate random code
        batch_codes = torch.from_numpy(
            np.random.RandomState(args.seed).randn(1, G.z_dim)).to(device)
        
        # Extract 3D shape
        if args.save_shape:
            print("Extracting 3D mesh...")
            max_batch = 1000000
            shape_res = args.shape_res

            voxel = create_voxel(N=shape_res, voxel_corner=[0, 0, 0], voxel_length=2.0)
            voxel_grid = voxel['voxel_grid'].to(device)
            densities = torch.zeros(voxel_grid.size(0), voxel_grid.size(1), 1).to(device)

            head = 0
            with tqdm.tqdm(total=voxel_grid.size(1)) as pbar:
                while head < voxel_grid.size(1):
                    density = G.sample(
                        voxel_grid[:, head:head + max_batch], batch_codes, sat_img, **G_kwargs)['density']
                    densities[:, head:head + max_batch, :] = density
                    head += max_batch
                    pbar.update(max_batch)
            densities = densities.reshape(shape_res, shape_res, shape_res).cpu().numpy()

            import mcubes
            vertices, faces = mcubes.marching_cubes(densities, 5)
            vertices = vertices / (shape_res - 1) * 2 - 1
            
            # Query vertex colors
            vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device).unsqueeze(0)
            vertices_colors = G.sample(vertices_tensor, batch_codes, sat_img, **G_kwargs)['color']
            vertices_colors = (vertices_colors * 255).squeeze(0).cpu().numpy().astype(np.uint8)
            vertices = vertices[:, [1, 2, 0]]

            save_obj(vertices, faces, vertices_colors[...,:3], os.path.join(work_dir, f'mesh.obj'))

        # Generate tri-plane features
        wp = G.mapping(batch_codes, None, sat_img)
        triplanes = G.synthesis_tri_plane(wp, sat_img=sat_img)
        
        # Sky mapping
        w_sky = G.sky_img_mapping(batch_codes, histo_sky,
                                 truncation_psi=1,
                                 truncation_cutoff=None,
                                 update_emas=False)
        
        # Render street view at single position
        if args.save_sky or args.save_street:
            gen_output = G.from_triplane_2_grd(wp, triplanes,
                                              sat_img=sat_img,
                                              w_sky=w_sky,
                                              update_emas=False,
                                              origin_H_W=position)
            results.update(gen_output)
            max_street_depth = max(gen_output.image_depth.max().item(), max_street_depth)
        
        # Render satellite view
        if args.save_sat:
            gen_output = G.from_triplane_2_sat(triplanes,
                                              sat_img=sat_img,
                                              random_crop=False,
                                              w=None)
            results.sat_img_raw = gen_output.image_raw
            results.sat_dep = gen_output.image_depth
        
        # Generate video
        if args.save_video:
            print("Generating video...")
            output_vid = []
                
            for i, position_i in enumerate(position_vid):
                position_input = torch.from_numpy(np.array(position_i)).unsqueeze(0).float().cuda()
                gen_output = G.from_triplane_2_grd(wp, triplanes,
                                                  sat_img=sat_img,
                                                  w_sky=w_sky,
                                                  update_emas=False,
                                                  origin_H_W=position_input)
                output_vid.append([gen_output.image, gen_output.image_depth, gen_output.alpha_raw])
                max_street_depth = max(gen_output.image_depth.max().item(), max_street_depth)

    # Save results
    if args.save_sat:
        print("Saving satellite-view results...")
        result_sat_img = results.sat_img_raw
        result_sat_img = result_sat_img.clamp(-1, 1)*0.5 +0.5
        torchvision.utils.save_image(result_sat_img, os.path.join(work_dir, 'pred_satrgb.png'))

        result_sat_dep = results.sat_dep
        result_sat_dep = 1- (result_sat_dep- result_sat_dep.min())/(2-result_sat_dep.min())
        result_sat_dep = result_sat_dep.clamp(0, 1)*255
        result_sat_dep = result_sat_dep.squeeze().cpu().numpy().astype(np.uint8)
        result_sat_dep = cv2.applyColorMap(result_sat_dep, cv2.COLORMAP_PLASMA)
        cv2.imwrite(os.path.join(work_dir, 'pred_satdep.png'), result_sat_dep)

        # Save satellite input with trajectory
        sat_img_input = sat_img[0].clamp(-1, 1)*0.5 +0.5
        size = sat_img_input.size(1)
        sat_img_input_to_draw = to_pil(sat_img_input.clone())
        draw = ImageDraw.Draw(sat_img_input_to_draw)
            
        if args.save_video:
            num_positions = len(position_vid)
            for idx in range(num_positions):
                start_pos = position_vid[idx]
                x1, y1 = start_pos[1] * size / 2 + size / 2, start_pos[0] * size / 2 + size / 2
                
                if idx != num_positions - 1:
                    end_pos = position_vid[idx + 1]
                    x2, y2 = end_pos[1] * size / 2 + size / 2, end_pos[0] * size / 2 + size / 2
                    factor = idx / (num_positions - 1)
                    color = interpolate_color(factor)
                    draw.line((x1, y1, x2, y2), fill=color, width=2)

        sat_img_input_to_draw = torchvision.transforms.ToTensor()(sat_img_input_to_draw)
        torchvision.utils.save_image(sat_img_input_to_draw, os.path.join(work_dir, 'sat_input.png'))

        sat_img_input_to_draw = to_pil(sat_img_input.clone())
        draw = ImageDraw.Draw(sat_img_input_to_draw)
        for i in position:
            x, y, radius = i[1]*size/2 + size/2, i[0]*size/2 + size/2, 1
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='red')
            draw.line((x, y - 15, x, y), fill='red', width=2)
            draw.line((x - 15, y, x, y), fill='green', width=2)
            draw.line((x, y, x + 15, y), fill='blue', width=2)
            draw.line((x, y, x, y + 15), fill='yellow', width=2)
        sat_img_input_to_draw = torchvision.transforms.ToTensor()(sat_img_input_to_draw)
        torchvision.utils.save_image(sat_img_input_to_draw, os.path.join(work_dir, 'sat_input_position.png'))
        
    if args.save_sky:
        print("Saving sky results...")
        result_sky_img = results.sky_img
        result_sky_img = result_sky_img.clamp(-1, 1)*0.5 +0.5
        torchvision.utils.save_image(result_sky_img, os.path.join(work_dir, 'pred_sky.png'))
        
    if args.save_street:
        print("Saving street-view results...")
        result_street_img = results.image
        result_street_img = result_street_img.clamp(-1, 1)*0.5 +0.5
        result_street_img = result_street_img.mul(255).squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
        result_street_img = cv2.cvtColor(result_street_img, cv2.COLOR_RGB2BGR)

        gray_image = results.image_depth
        gray_image = gray_image / max_street_depth
        gray_image = gray_image.clamp(0, 1)*255
        gray_image = gray_image.squeeze().cpu().numpy().astype(np.uint8)
        result_street_dep = cv2.applyColorMap(gray_image, cv2.COLORMAP_PLASMA)
        opacity = results.alpha_raw.squeeze().cpu().numpy() < 0.5
        save_opacity = results.alpha_raw.squeeze().cpu().mul(255).numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(work_dir, 'pred_street_opacity.png'), save_opacity)

        result_street_dep[opacity] = [0, 0, 0]
        shape = result_street_dep.shape
        result_street_dep = cv2.resize(result_street_dep, (shape[1]*2, shape[0]*2), interpolation=cv2.INTER_LINEAR)

        result_street = np.concatenate([result_street_img, result_street_dep], axis=0)
        cv2.imwrite(os.path.join(work_dir, 'pred_street.png'), result_street)
        cv2.imwrite(os.path.join(work_dir, 'pred_street_img.png'), result_street_img)
        cv2.imwrite(os.path.join(work_dir, 'pred_street_dep.png'), result_street_dep)

    if args.save_video:
        print("Saving video...")
        if os.path.exists(os.path.join(work_dir, 'save_vid')):
            os.system('rm -rf '+os.path.join(work_dir, 'save_vid'))
            os.system('rm -rf '+os.path.join(work_dir, 'save_street_vid'))
            os.system('rm -rf '+os.path.join(work_dir, 'save_street_only_vid'))
            os.system('rm -rf '+os.path.join(work_dir, 'save_sat'))
        os.makedirs(os.path.join(work_dir, 'save_vid'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'save_street_vid'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'save_street_only_vid'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'save_sat'), exist_ok=True)

        for i in range(len(output_vid)):
            position = position_vid[i]
            save_name = str('{:03d}'.format(i)) + '.png'
            result_street_img = output_vid[i][0]
            result_street_img = result_street_img.clamp(-1, 1)*0.5 +0.5
            result_street_img = result_street_img.mul(255).squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
            result_street_img = cv2.cvtColor(result_street_img, cv2.COLOR_RGB2BGR)

            gray_image = output_vid[i][1]
            gray_image = gray_image / max_street_depth
            gray_image = gray_image.clamp(0, 1)*255
            gray_image = gray_image.squeeze().cpu().numpy().astype(np.uint8)
            result_street_dep = cv2.applyColorMap(gray_image, cv2.COLORMAP_PLASMA)
            opacity = output_vid[i][2].squeeze().cpu().numpy() < 0.5
            result_street_dep[opacity] = [0, 0, 0]
            shape = result_street_dep.shape
            result_street_dep = cv2.resize(result_street_dep, (shape[1]*2, shape[0]*2), interpolation=cv2.INTER_LINEAR)
            result_street = np.concatenate([result_street_img, result_street_dep], axis=0)

            if args.visual_direction:
                result_street = cv2.cvtColor(result_street, cv2.COLOR_RGB2BGR)
                result_street_to_draw = to_pil(result_street)
                draw = ImageDraw.Draw(result_street_to_draw)
                draw.line((256, 0, 256, 128), fill='red', width=1)
                draw.line((384, 0, 384, 128), fill='blue', width=1)
                draw.line((128, 0, 128, 128), fill='green', width=1)
                draw.line((0, 0, 0, 128), fill='yellow', width=1)
                draw.line((511, 0, 511, 128), fill='yellow', width=1)
                result_street = np.array(result_street_to_draw)
                result_street = cv2.cvtColor(result_street, cv2.COLOR_RGB2BGR)

            size = sat_img_input.size(1)
            sat_img_input_to_draw = to_pil(sat_img_input.clone())
            draw = ImageDraw.Draw(sat_img_input_to_draw)
            x, y, radius = position_vid[i][1]*size/2 + size/2, position_vid[i][0]*size/2 + size/2, 3
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='green')

            line_length = 15
            draw.line((x, y - line_length, x, y), fill='red', width=2)
            draw.line((x - line_length, y, x, y), fill='green', width=2)
            draw.line((x, y, x + line_length, y), fill='blue', width=2)
            draw.line((x, y, x, y + line_length), fill='yellow', width=2)

            sat_img_input_to_draw = np.array(sat_img_input_to_draw)
            sat_img_input_to_draw = cv2.cvtColor(sat_img_input_to_draw, cv2.COLOR_RGB2BGR)
            result_cat = np.concatenate([sat_img_input_to_draw, result_street], axis=1)
            cv2.imwrite(os.path.join(work_dir, 'save_sat', save_name), sat_img_input_to_draw)
            cv2.imwrite(os.path.join(work_dir, 'save_street_vid', save_name), result_street)
            cv2.imwrite(os.path.join(work_dir, 'save_vid', save_name), result_cat)
            cv2.imwrite(os.path.join(work_dir, 'save_street_only_vid', save_name), result_street_img)
        
        cv2.imwrite(os.path.join(work_dir,'sat_list.png'),
                   cv2.cvtColor(sat_img_input.cpu().mul(255).numpy().squeeze().transpose(1,2,0).astype(np.uint8), 
                               cv2.COLOR_RGB2BGR))

        # Generate videos using ffmpeg
        original_dir = os.getcwd()
        os.chdir(work_dir)
        os.system('ffmpeg -y -framerate 5 -pattern_type glob -i "save_vid/*.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ./vid.gif\n')
        os.system('ffmpeg -y -framerate 5 -pattern_type glob -i "save_vid/*.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ./vid.mp4\n')
        os.system('ffmpeg -y -framerate 5 -pattern_type glob -i "save_street_vid/*.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ./vid_str.mp4\n') 
        os.system('ffmpeg -y -framerate 5 -pattern_type glob -i "save_sat/*.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ./sat.mp4\n')
        os.system('ffmpeg -y -framerate 5 -pattern_type glob -i "save_street_only_vid/*.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p ./save_street_only_vid.mp4\n') 
        os.chdir(original_dir)

    # Save sky reference image
    sky_img = Image.open(args.sky_path).convert('RGB').resize((512, 128))
    sky_img.save(os.path.join(work_dir, 'gt_sky_rgb.png'))
    if masked_sky_img is not None:
        masked_sky_img = masked_sky_img.transpose(1,2,0)
        masked_sky_img = cv2.cvtColor(masked_sky_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(work_dir, 'gt_sky_masked.png'), (masked_sky_img*255).astype(np.uint8))

    print(f"\nResults saved to: {work_dir}")

if __name__ == '__main__':
    main()
