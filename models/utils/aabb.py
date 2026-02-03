import numpy as np
import torch
from typing import Literal, Tuple
# borrowed from https://github.com/nerfstudio-project/nerfstudio
from einops import rearrange

def intersect_aabb(
    origins: torch.Tensor,
    directions: torch.Tensor,
    aabb: torch.Tensor = torch.tensor([-1., -1., -1., 1., 1., 1.]).float(),
    max_bound: float = 1e10,
    invalid_value: float = 1e10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of ray intersection with AABB box

    Args:
        origins: [N,3] tensor of 3d positions
        directions: [N,3] tensor of normalized directions
        aabb: [6] array of aabb box in the form of [x_min, y_min, z_min, x_max, y_max, z_max]
        max_bound: Maximum value of t_max
        invalid_value: Value to return in case of no intersection

    Returns:
        t_min, t_max - two tensors of shapes N representing distance of intersection from the origin.
    """
    # send aabb to origins's device
    if len(origins.shape) == 4:
        origins = rearrange(origins, 'b h w c -> (b h w) c')
        directions = rearrange(directions, 'b h w c -> (b h w) c')
    aabb = aabb.to(origins.device)
    B = max(origins.size(0), directions.size(0))
    tx_min = (aabb[:3] - origins) / directions
    tx_max = (aabb[3:] - origins) / directions
    
    t_min = torch.stack((tx_min, tx_max)).amin(dim=0)
    t_max = torch.stack((tx_min, tx_max)).amax(dim=0)

    t_min = t_min.amax(dim=-1)
    t_max = t_max.amin(dim=-1)

    t_min = torch.clamp(t_min, min=0, max=max_bound)
    t_max = torch.clamp(t_max, min=0, max=max_bound)

    cond = t_max <= t_min
    ### fix a bug may caused by version change?
    invalid_value = torch.tensor(invalid_value).repeat(B).float().to(origins.device)
    t_min = torch.where(cond, invalid_value, t_min)
    t_max = torch.where(cond, invalid_value, t_max)
    # tmax is what I need
    return t_min, t_max

def intersect_aabb_end(origin,dir,min=0,max=4):
    t_max = intersect_aabb(origin,dir)[1]
    assert torch.isnan(t_max).any() == False , "nan in t_max of intersect_aabb_end"
    assert min < t_max.min() < max, "t_max out of range %s, min is %s, max is %s" % (t_max.min(), min, max)
    return t_max

if __name__ == '__main__':
    # Example usage:
    origin =  torch.tensor([[-0.3, 0, 0], [-1, -1, -1]]).float() # Nx3 origins
    dir = torch.tensor([[1, 0, 0], [0, 0, 1]]).float()    # Nx3 directions
    # aabb = torch.tensor([-1., -1., -1., 1., 1., 1.]).float()

    distances = intersect_aabb_end(origin, dir)
    print(distances)
