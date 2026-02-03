import numpy as np
import torch,math
from PIL import Image
import torchvision
from easydict import EasyDict as edict
from einops import repeat, rearrange
from .rendering.point_sampler import perturb_points_per_ray
from models.utils.aabb import intersect_aabb_end
import torch.nn.functional as F
import torch.nn as nn
import random


def position_produce(opt): 
    depth_channel =  opt.arch.gen.depth_arch.output_nc 
    if  opt.optim.ground_prior:
        depth_channel = depth_channel+1
    z_ = torch.arange(depth_channel)/depth_channel
    x_ = torch.arange(opt.data.sat_size[1])/opt.data.sat_size[1]
    y_ = torch.arange(opt.data.sat_size[0])/opt.data.sat_size[0]
    Z,X,Y = torch.meshgrid(z_,x_,y_)
    input = torch.cat((Z[...,None],X[...,None],Y[...,None]),dim=-1).to(opt.device)
    pos = positional_encoding(opt,input)
    pos = pos.permute(3,0,1,2)
    return  pos

def positional_encoding(opt,input): # [B,...,N]
    shape = input.shape
    freq = 2**torch.arange(opt.arch.gen.PE_channel,dtype=torch.float32,device=opt.device)*np.pi # [L]
    spectrum = input[...,None]*freq # [B,...,N,L]
    sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
    input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
    input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
    return input_enc



def get_original_coord(W,H,full=True):
    '''
    W: width of pano
    H: height of pano
    if dataset is CVACT, ful=True, return the original coordinate of CVACT
    if dataset is CVUSA, ful=False,
    fill = False only used for CVUSA dataset
    '''
    _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T

    ## simplify for new framework
    if full:
        _theta = (1 - 2 * (_x) / H) * np.pi/2 # latitude 
    else:   ### no 
        _theta = (1 - 2 * (_x) / H) * np.pi/4

    _phi = math.pi* ( 1 -2* (_y)/W ) # longtitude 
    _phi = math.pi*( - 0.5 - 2* (_y)/W )
    axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
    axis1 = np.sin(_theta).reshape(H, W, 1) 
    axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1) 
    pano_direction = np.concatenate((axis0, axis1, axis2), axis=2)
    return pano_direction  

class Point_sampler_pano(torch.nn.Module):
    def __init__(self,
                pano_direction,
                sample_total_length=None,
                origin_height=2,
                realworld_scale=30,
                max_height=20,
                num_points=300,
                perturbation_strategy = 'uniform',
                aabb_strict =False,
                data_type = 'cvact',
                ):
        super().__init__()
        if data_type == 'vigor':
            self.sample_total_length = np.sqrt(1.5**2+1.5**2+1.9**2)
        elif data_type.lower() in ['cvact','cvusa']:
        # the next two row not used in vigor
            self.realworld_scale = realworld_scale
            self.sample_total_length =  max(np.sqrt((2*((realworld_scale/2)**2))+(2)**2), \
                                    np.sqrt(2*((realworld_scale/2)**2)+(max_height-origin_height)**2))*2/realworld_scale if  sample_total_length is None else sample_total_length
            self.origin_z = torch.ones(1)*(-1+(origin_height/(realworld_scale/2))) ### -1 is the loweast position in regular cooridinate
        else:
            raise NotImplementedError
        

        self.pano_direction = pano_direction
        self.num_points = num_points
        if not aabb_strict:
            self.sample_len = ((torch.arange(self.num_points)+1)*(self.sample_total_length/self.num_points)).cuda().float()

        self.voxel_low = -1
        self.voxel_max = 1

        self.perturbation_strategy = perturbation_strategy
        self.aabb_strict = aabb_strict

    def forward(self,
                batch_size,
                origin_H_W=None,
                **kwargs):
        if origin_H_W is None: ### origin_H_W is the photo taken space in regular coordinate
            origin_z = self.origin_z.repeat(batch_size,1)
            origin_H,origin_w = torch.zeros([batch_size,1]),torch.zeros([batch_size,1])   
        elif origin_H_W.shape[1] == 2:
            origin_z = self.origin_z.repeat(batch_size,1)
            origin_H,origin_w = origin_H_W[:,0].unsqueeze(1), origin_H_W[:,1].unsqueeze(1)
        elif origin_H_W.shape[1] == 3:
            origin_H, origin_w, origin_z = origin_H_W[:,0].unsqueeze(1), origin_H_W[:,1].unsqueeze(1), origin_H_W[:,2].unsqueeze(1)
        # change it to the dataloader for better understanding
        # if none, keep the origin positin, else, change to dataloader
        origin = torch.cat([origin_w,origin_z,origin_H],dim=1).cuda()[:,None,None,:]  ## w,z,h, samiliar to NERF coordinate definition
        origin = origin[...,None]
        pano_direction = self.pano_direction[...,None]


        H,W = pano_direction.shape[1],pano_direction.shape[2]

        output = edict()
        output.rays_world = repeat(pano_direction, '1 h w c 1 -> b h w c', b=batch_size )[...,[0,2,1]]  # (1,h,w,3,1) -> (batch_size, h, w, 3)

        output.ray_origins = rearrange(origin, 'b 1 1 c 1 -> b c')[...,[0,2,1]]



        if self.aabb_strict:
            #  from b c to (b h w) c
            origin_for_aabb = repeat(output.ray_origins, 'b c -> b h w c', h = H, w = W)
            # from  b h w c to (b h w) c
            # pano_direction_for_aabb = repeat(output.rays_world, 'b h w c -> b h w c', h = H, w = W)
            sample_total_length = intersect_aabb_end(origin_for_aabb,output.rays_world,min=0,max=self.sample_total_length)
            sample_total_length = rearrange(sample_total_length, '(b h w) -> b h w 1', b=batch_size, h = H, w = W )
            output.radii_raw = (torch.arange(self.num_points)+1)[None,None,None,:].to(sample_total_length.device) * (sample_total_length/self.num_points)

            # from (b h w) to  (b h w) k
            
        else:
            depth = self.sample_len[None,None,None,None,:]
            output.radii_raw  = repeat(depth, '1 1 1 1 k -> b h w k', b=batch_size, h = H, w = W )  # (1,h,w,1,k) -> (batch_size, h, w, k)
        output.radii =  perturb_points_per_ray(output.radii_raw,strategy=self.perturbation_strategy)
        # sample_point = origin + pano_direction * depth
        sample_point = origin + pano_direction * output.radii.unsqueeze(-2)
        
        grid = sample_point.permute(0,4,1,2,3)[...,[0,2,1]]
        # grid[...,2]   = ((grid[...,2]-self.voxel_low)/(self.voxel_max-self.voxel_low))*2-1
        # grid = grid.float()
        
        output.points_world =  rearrange(grid, 'b k h w c -> b h w k c')
        return  output


def get_sat_ori(resolution):
    y_range =  (torch.arange(resolution,dtype=torch.float32,)+0.5)/(0.5*resolution)-1
    x_range = (torch.arange(resolution,dtype=torch.float32,)+0.5)/(0.5*resolution)-1
    Y,X = torch.meshgrid(y_range,x_range)
    Z = torch.ones_like(Y)
    xy_grid = torch.stack([X,Z,Y],dim=-1)[None,:,:]
    return xy_grid


class Point_sampler_ortho(torch.nn.Module):
    '''
    point sampler designed for ortho image,


    '''
    def __init__(self,
                 num_points,
                 resolution=128,
                 perturbation_strategy = 'uniform',
                 ):
        super().__init__()
        self.perturbation_strategy = perturbation_strategy
        # not used any more
        self.resolution = resolution
        self.sat_ori = get_sat_ori(256)[...,None].cuda()
        self.sat_dir  = torch.tensor([0,-1,0])[None,None,None,:,None].cuda()
        self.sample_total_length = 2
        self.num_points = num_points
        self.sample_len = ((torch.arange(self.num_points)+1)*(self.sample_total_length/self.num_points)).cuda().float()

    def forward(self,
                batch_size,
                random_crop=True):
        depth = self.sample_len[None,None,None,None,:]
        # sample_point = self.sat_ori + self.sat_dir * depth
        # grid = sample_point.permute(0,4,1,2,3)[...,[0,2,1]]
        output = edict()
        if random_crop:
            render_pixel = 128
            start_h = random.randint(0,256-render_pixel-1)
            start_w = random.randint(0,256-render_pixel-1)
            output.idx = torch.tensor([start_h,start_w]).cuda()
            sat_ori = self.sat_ori[:,start_h:start_h+render_pixel,start_w:start_w+render_pixel,:]
        else:
            sat_ori = self.sat_ori
            render_pixel = 256
        
        # output.points_world =  repeat(grid, '1 k h w c -> b h w k c', b=batch_size)
        output.rays_world = repeat(self.sat_dir, '1 1 1 c 1 -> b h w c', b=batch_size,  h = render_pixel, w = render_pixel )[...,[0,2,1]]  # (1,h,w,3,1) -> (batch_size, h, w, 3)
        output.radii_raw  = repeat(depth, '1 1 1 1 k -> b h w k', b=batch_size, h = render_pixel, w = render_pixel )  # (1,h,w,1,k) -> (batch_size, h, w, k, 1)
        output.ray_origins = repeat(sat_ori, '1 h w c 1 -> b h w c',b=batch_size)[...,[0,2,1]]
        output.radii =  perturb_points_per_ray(output.radii_raw,strategy=self.perturbation_strategy)

        sample_point = sat_ori + self.sat_dir * output.radii.unsqueeze(-2)
        
        grid = sample_point.permute(0,4,1,2,3)[...,[0,2,1]]
        # grid[...,2]   = ((grid[...,2]-self.voxel_low)/(self.voxel_max-self.voxel_low))*2-1
        # grid = grid.float()
        output.points_world =  rearrange(grid, 'b k h w c -> b h w k c')
        return  output





class RGB_Reprerenter(torch.nn.Module):
    def __init__(self,
                ):
        super().__init__()

    def forward(self,
                points,
                image,
                ):
        point_h_w = points[...,0:2].unsqueeze(2) # b, N, 1, 2
        rgb_feature = F.grid_sample(image,point_h_w).squeeze(-1).permute(0,2,1) # b, C, N, 1 -> b, N, C
        return rgb_feature
