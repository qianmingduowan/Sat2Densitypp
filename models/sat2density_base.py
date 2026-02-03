import numpy as np
from torch.nn import Upsample as NearestUpsample
from functools import partial

import sys 
sys.path.append(".") 
from .sat2density_base_layer import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.eg3d_model_helper import Conv2dLayer, FullyConnectedLayer
from models.utils.eg3d_model_helper import DiscriminatorBlock,SynthesisBlock
from utils import eg3d_misc as misc
import cv2
from easydict import EasyDict as edict



class StyleMLP(nn.Module):
    r"""MLP converting style code to intermediate style representation."""

    def __init__(self, style_dim, out_dim, hidden_channels=256, leaky_relu=True, num_layers=5, normalize_input=True,
                 output_act=True):
        super(StyleMLP, self).__init__()

        self.normalize_input = normalize_input
        self.output_act = output_act
        fc_layers = []
        fc_layers.append(FullyConnectedLayer(style_dim, hidden_channels, bias=True,activation='lrelu'))
        # fc_layers.append(nn.Linear(style_dim, hidden_channels, bias=True))
        for i in range(num_layers-1):
            fc_layers.append(FullyConnectedLayer(hidden_channels, hidden_channels, bias=True,activation='lrelu'))
        self.fc_layers = nn.ModuleList(fc_layers)
        activation = 'lrelu' if output_act==True  else 'linear'
        self.fc_out = FullyConnectedLayer(hidden_channels, out_dim, bias=True, activation = activation)

        # if leaky_relu:
        #     self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # else:
        #     self.act = partial(F.relu, inplace=True)

    def forward(self, z):
        r""" Forward network

        Args:
            z (N x style_dim tensor): Style codes.
        """
        if self.normalize_input:
            z = F.normalize(z, p=2, dim=-1,eps=1e-6)
        for fc_layer in self.fc_layers:
            z = fc_layer(z)
        z = self.fc_out(z)
        # if self.output_act:
            # z = self.act(z)
        return z

class histo_process_v2(nn.Module):
    r"""Histo process to replace Style Encoder constructor.

    Args:
        style_enc_cfg (obj): Style encoder definition file.
    """
    def __init__(self,style_enc_cfg):
        super().__init__()
        self.style_dims = style_enc_cfg.style_dims
        self.no_vae = getattr(style_enc_cfg, 'no_vae', False)
        num_filters = getattr(style_enc_cfg, 'num_filters', 180)
        self.layer1 = FullyConnectedLayer(style_enc_cfg.input_image_channels,num_filters,activation='lrelu')
        self.layer4 = FullyConnectedLayer(num_filters, num_filters)

        # self.layer1 = LinearBlock(style_enc_cfg.input_image_channels,num_filters)
        # self.layer4 = LinearBlock(num_filters, num_filters)
        self.fc_mu = FullyConnectedLayer(num_filters, self.style_dims,bias=False)
        # self.fc_mu = LinearBlock(num_filters, style_dims,nonlinearity='tanh')
        self.fc_var = FullyConnectedLayer(num_filters, self.style_dims,bias=False)
            # self.fc_var = LinearBlock(num_filters, style_dims,nonlinearity='tanh')


    def forward(self,histo):
        x = self.layer1(histo)
        x = self.layer4(x)
        mu = self.fc_mu(x) #[-1,1]
        # tanh
        z = torch.tanh(mu)
        return  z

class histo_process(nn.Module):
    r"""Histo process to replace Style Encoder constructor.

    Args:
        style_enc_cfg (obj): Style encoder definition file.
    """
    def __init__(self,style_enc_cfg):
        super().__init__()
        style_dims = style_enc_cfg.style_dims
        self.no_vae = getattr(style_enc_cfg, 'no_vae', False)
        num_filters = getattr(style_enc_cfg, 'num_filters', 180)
        self.layer1 = FullyConnectedLayer(style_enc_cfg.input_image_channels,num_filters,activation='lrelu')
        self.layer4 = FullyConnectedLayer(num_filters, num_filters)

        # self.layer1 = LinearBlock(style_enc_cfg.input_image_channels,num_filters)
        # self.layer4 = LinearBlock(num_filters, num_filters)
        self.fc_mu = FullyConnectedLayer(num_filters, style_dims,bias=False)
        # self.fc_mu = LinearBlock(num_filters, style_dims,nonlinearity='tanh')
        if not self.no_vae:
            self.fc_var = FullyConnectedLayer(num_filters, style_dims,bias=False)
            # self.fc_var = LinearBlock(num_filters, style_dims,nonlinearity='tanh')


    def forward(self,histo):
        x = self.layer1(histo)
        x = self.layer4(x)
        mu = self.fc_mu(x) #[-1,1]
        # tanh
        mu = torch.tanh(mu)
        if not self.no_vae:
            logvar = self.fc_var(x) # [-1,1]
            logvar = torch.tanh(logvar)
            std = torch.exp(0.5 * logvar)  # [0.607,1.624]
            eps = torch.randn_like(std) 
            z = eps.mul(std) + mu
        else:
            z = mu
            logvar = torch.zeros_like(mu)
        return mu, logvar, z


class inner_Generator_split(nn.Module):
    r"""Pix2pixHD coarse-to-fine generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg,inner_cfg,num_input_channels=3):
        super().__init__()

        # pix2pixHD has a global generator.
        # By default, pix2pixHD using instance normalization.

        base_res_block = partial(Res2dBlock, order='CC')
        
        # Know what is the number of available segmentation labels.

        # Global generator model.

        num_out_put_channels = getattr(inner_cfg, 'output_nc', 64)
        num_filters = getattr(inner_cfg, 'num_filters', 64)
        num_downsamples = 4
        num_res_blocks = getattr(inner_cfg, 'num_res_blocks', 9)
        # First layer.

        model = [
                Conv2dLayer(num_input_channels,num_filters,kernel_size=3,activation='lrelu')
                ]
        # model = [base_conv_block(num_input_channels, num_filters,
        #                          kernel_size=7, padding=3)]
        # model += [nn.PReLU()]

        # Downsample.
        for i in range(num_downsamples):
            ch = num_filters * (2 ** i)
            model += [Conv2dLayer(ch,ch*2,kernel_size=3,activation='lrelu',down=2)]
        # ResNet blocks.
        ch = num_filters * (2 ** num_downsamples)
        for i in range(num_res_blocks):
            model += [base_res_block(ch, ch, 3)]
        self.model = nn.Sequential(*model)
        # Upsample.
        assert num_downsamples == 4
        if not (inner_cfg.name =='render' and 'style_inject' in gen_cfg.keys()):
            list = [16,8,4,2]
        else:
            list = [16,6,6,6]

        self.up0 = Conv2dLayer(num_filters * list[0], num_filters*list[1], 3, activation='lrelu',up=2)
        self.up1 = Conv2dLayer(num_filters * list[1], num_filters*list[2], 3,up=2)
        self.up2 = Conv2dLayer(num_filters * list[2], num_filters*list[3], 3,up=2)
        self.up3 = Conv2dLayer(num_filters * list[3], num_filters,         3, activation='lrelu', up=2)
        self.up_end = Conv2dLayer(num_filters, num_out_put_channels, 3)
        if inner_cfg.name =='render' and 'style_inject' in gen_cfg.keys():
            self.fc_z_cond = FullyConnectedLayer(gen_cfg.style_enc_cfg.interm_style_dims, 4* list[-1] * num_filters)
            # self.fc_z_cond = nn.Linear(style_dim, 4* list[-1] * num_filters)

    def modulate(self, x, w, b):
        w = w[..., None, None]
        b = b[..., None, None]
        return x * (w+1) + b

    def forward(self, input,z=None):
        r"""Coarse-to-fine generator forward.

        Args:
            input (4D tensor) : Input semantic representations.
        Returns:
            output (4D tensor) : Synthesized image by generator.
        """
        if z is not None:
            z = self.fc_z_cond(z)
            adapt = torch.chunk(z, 2 * 2, dim=-1)
        input = self.model(input)
        input = self.up0(input)
        input = self.up1(input)
        if z is not None:
            input = self.modulate(input, adapt[0], adapt[1])
        input = F.leaky_relu(input,negative_slope=0.2)
        input = self.up2(input)
        if z is not None:
            input = self.modulate(input, adapt[2], adapt[3])
        input = F.leaky_relu(input,negative_slope=0.2)
        input = self.up3(input)
        input = self.up_end(input)
        return input



class DenoiseNet(nn.Module):
    r"""Pix2pixHD coarse-to-fine generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self,num_input_channels,use_noise):
        super().__init__()
        self.encoder_num_filters = [64,128,256,512]
        self.decoder_num_filters = self.encoder_num_filters[::-1]
        self.encoder_block_resolutions  = [128,64,32,16]
        # each num *2 in self.decoder_block_resolutions
        self.decoder_block_resolutions = [int(res*2) for res in self.encoder_block_resolutions[::-1]]
        self.dec_layer_kwargs = edict()
        self.dec_layer_kwargs.use_noise = use_noise
        self.w_dim = 512
        cur_layer_idx = 0
        self.num_ws = 0
        for i,res in enumerate(self.encoder_block_resolutions):
            in_channels = 0 if i == 0 else self.encoder_num_filters[i-1]
            temp_channels = self.encoder_num_filters[1] if i == 0 else self.encoder_num_filters[i-1] 
            out_channels = self.encoder_num_filters[i]
            use_fp16 = False
            # is_last = False
            block =  DiscriminatorBlock(in_channels,
                                        temp_channels,
                                        out_channels,
                                        resolution=res,
                                        img_channels=num_input_channels,
                                        first_layer_idx=cur_layer_idx,
                                        use_fp16=use_fp16,
                                        )
            setattr(self, f'Enc_{res}', block)
            # cur_layer_idx += block.num_conv

        for i,res in enumerate(self.decoder_block_resolutions):
            in_channels = self.decoder_num_filters[i]
            out_channels = self.decoder_num_filters[i+1] if i < len(self.decoder_num_filters)-1 else 3
            use_fp16 = False
            is_last = True if i == len(self.decoder_block_resolutions)-1 else False
            block =  SynthesisBlock(in_channels,
                                    out_channels,
                                    w_dim=self.w_dim,
                                    resolution=res,
                                    img_channels=3,
                                    first_layer_idx=cur_layer_idx,
                                    use_fp16=use_fp16,
                                    is_last=is_last,
                                    **self.dec_layer_kwargs)
            cur_layer_idx += block.num_conv
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'Dec_{res}', block)
        
    def forward(self, img, ws, update_emas=False, **block_kwargs):
        block_ws = []
        ## encoder
        x = None
        for res in self.encoder_block_resolutions:
            block = getattr(self, f'Enc_{res}')
            x, img  = block(x,img, **block_kwargs)
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.decoder_block_resolutions:
                block = getattr(self, f'Dec_{res}')
                block_ws.append(
                    ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        img = None
        for res, cur_ws in zip(self.decoder_block_resolutions, block_ws):
            block = getattr(self, f'Dec_{res}')
            x,img = block(x, img,cur_ws, **block_kwargs)
        return img
            
    

    # def forward(self, input,z=None):
    #     r"""Coarse-to-fine generator forward.

    #     Args:
    #         input (4D tensor) : Input semantic representations.
    #     Returns:
    #         output (4D tensor) : Synthesized image by generator.
    #     """
    #     if z is not None:
    #         z = self.fc_z_cond(z)
    #         adapt = torch.chunk(z, 2 * 2, dim=-1)
    #     input = self.model(input)
    #     input = self.up0(input)
    #     input = self.up1(input)
    #     if z is not None:
    #         input = self.modulate(input, adapt[0], adapt[1])
    #     input = F.leaky_relu(input,negative_slope=0.2)
    #     input = self.up2(input)
    #     if z is not None:
    #         input = self.modulate(input, adapt[2], adapt[3])
    #     input = F.leaky_relu(input,negative_slope=0.2)
    #     input = self.up3(input)
    #     input = self.up_end(input)
    #     return input


class Equirectangular():
    """
    Random sample a panorama image into a perspective view
    take https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py as a reference
    """
    def __init__(self, width = 256, height = 256, FovX = 100, theta = [0, 0]):
        """
        width: output image's width
        height: output image's height
        FovX: perspective camera FOV on x-axis (degree)
        theta: theta field where img's theta degree from 
        """
        self.theta = theta
        self.width = width
        self.height = height
        self.type = type

        #create x-axis coordinates and corresponding y-axis coordinates
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y) 
        
        #create homogenerous coordinates
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        
        #translation matrix
        f = 0.5 * width * 1 / np.tan(np.radians(FovX/2))
        # cx = (width - 1) / 2.0
        # cy = (height - 1) / 2.0
        cx = (width) / 2.0
        cy = (height) / 2.0        
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        xyz = xyz @ K_inv.T
        self.xyz = xyz  ### self.xyz is the direction of the each ray in the camera space when camera is fixed



    def __call__(self, batch_size): 
        # batch = img1.shape[0]
        PHI, THETA = self.getRandomRotation(batch_size)
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        #rotation matrix
        xy_grid = []
        for i in range(batch_size):
            R1, _ = cv2.Rodrigues(y_axis * np.radians(PHI[i]))
            R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(THETA[i]))
            R = R2 @ R1
            #rotate
            xyz = self.xyz @ R.T  ### ### xyz is the direction of the each ray in the camera space when camera is rotate
            norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
            xyz_norm = xyz / norm
            
            #transfer to image coordinates
            xy = self.xyz2xy(xyz_norm)
            xy = torch.from_numpy(xy).cuda().unsqueeze(0)
            xy_grid.append(xy)
        xy = torch.cat(xy_grid,dim=0)

        #resample
        return xy

    def xyz2xy(self, xyz_norm):
        #normlize
        x = xyz_norm[..., 0]
        y = xyz_norm[..., 1]
        z = xyz_norm[..., 2]

        lon = np.arctan2(x, z)
        lat = np.arcsin(y)
        ### transfer to the lon and lat

        X = lon / (np.pi)
        Y = lat / (np.pi) * 2
        xy = np.stack([X, Y], axis=-1)
        xy = xy.astype(np.float32)
        
        return xy

    def getRandomRotation(self,batch_size):
        # phi = np.random.rand(batch_size) * 360 -180
        phi = np.random.randint(-180,180,batch_size)
        assert(self.theta[0]<self.theta[1])
        theta = np.random.randint(self.theta[0],self.theta[1],batch_size)
        # theta = np.random.rand(batch_size)*(self.theta[1]-self.theta[0])-self.theta[0]
        return phi, theta

    

if __name__=='__main__':
    from easydict import EasyDict as edict
    style_enc_cfg = edict()
    style_enc_cfg.histo.mode = 'RGB'
    style_enc_cfg.histo.num_filters = 180
    model = histo_process(style_enc_cfg)