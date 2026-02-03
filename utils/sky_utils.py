# python3.8
"""Utility functions for sky image processing."""

import numpy as np
import torch
from PIL import Image


def make_histo(str_path, to_tensor=True, return_sky_img=False):
    """Extract sky histogram from panorama image.
    
    Args:
        str_path: Path to panorama image
        to_tensor: Whether to return as tensor
        return_sky_img: Whether to also return the masked sky image
        
    Returns:
        Sky histogram (and optionally masked sky image)
    """
    grd_img_path = str_path
    grd_img = Image.open(grd_img_path).convert('RGB').resize((512, 128))
    grd_img = np.array(grd_img)
    grd_img = torch.from_numpy(grd_img).permute(2, 0, 1).unsqueeze(0).float()
    # Normalize to [-1,1]
    grd_img = grd_img / 255.0 * 2.0 - 1.0

    # Find corresponding sky mask
    if 'streetview/panos' in grd_img_path:
        mask_img_path = grd_img_path.replace('streetview/panos', 'sky_mask')
    else:
        if 'streetview' in grd_img_path:
            mask_img_path = grd_img_path.replace('streetview', 'pano_sky_mask')      
        elif 'panorama' in grd_img_path:
            mask_img_path = grd_img_path.replace('panorama', 'pano_sky_mask')
    mask_img_path = mask_img_path.replace('jpg', 'png') if mask_img_path.endswith('.jpg') else mask_img_path
    
    mask_img = Image.open(mask_img_path).convert('L').resize((512, 128))
    mask_img = np.array(mask_img)
    mask_img = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float() / 255.0
    # Round to 0 or 1
    mask_img = (mask_img > 0.5).float()

    sky_image = (grd_img + 1.) * mask_img
    sky_image = sky_image.detach().numpy()
    sky_histo = []
    
    for item in sky_image[0]:
        histo = np.histogram(item.flatten(), bins=100, range=(0, 2))[0]
        if histo.sum() == 0:
            continue
        histo = histo[10:]
        if histo.sum() != 0:
            histo = histo / histo.sum()
        sky_histo.append(histo)
    
    sky_histo = np.concatenate(sky_histo)
    
    if return_sky_img:
        masked_img = sky_image[0] * 0.5
        if to_tensor:
            return torch.from_numpy(sky_histo).unsqueeze(0).float(), masked_img
        else:
            return sky_histo, masked_img
    else:
        if to_tensor:
            return torch.from_numpy(sky_histo).unsqueeze(0).float()
        else:
            return sky_histo
