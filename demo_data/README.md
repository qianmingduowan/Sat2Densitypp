# Demo Data

This directory contains ready-to-use demo data for immediate testing without downloading full datasets.

## Contents

### CVACT
- **Satellite images**: 3 images in `satellite/`
- **Panorama images**: 3 matching images in `panorama/`
- **Sky masks**: 3 matching masks in `pano_sky_mask/`
- **Trajectory**: `pixels_aOIFiGzL6fvf0_YZIlyqFw_satView_polish.csv` (for `aOIFiGzL6fvf0_YZIlyqFw_satView_polish.png`)

### CVUSA
- **Satellite images**: 2 images in `satellite/`
- **Panorama images**: 2 matching images in `panorama/`
- **Sky masks**: 2 matching masks in `pano_sky_mask/`
- **Trajectory**: `pixels_0040836.csv` (for `0040836.jpg`)

### VIGOR
- **Satellite images**: 5 images in `satellite/`
- **Panorama images**: 4 matching images in `panorama/`
- **Sky masks**: 4 matching masks in `pano_sky_mask/`
- **Trajectory**: `pixels_satellite_41.88584553507432_-87.67181147737129.csv` (for `satellite_41.88584553507432_-87.67181147737129.png`)

## Important: Trajectory-Satellite Image Binding

⚠️ **Each trajectory CSV is specifically designed for its corresponding satellite image!**

Trajectories are drawn on specific satellite images based on the road layout, buildings, and obstacles in that scene. You **cannot** use a trajectory from one satellite image with a different satellite image.

### Images with Pre-generated Trajectories

**Ready to run (with trajectory)**:
- ✅ `demo_data/CVACT/satellite/aOIFiGzL6fvf0_YZIlyqFw_satView_polish.png`
- ✅ `demo_data/CVUSA/satellite/0040836.jpg`
- ✅ `demo_data/VIGOR/satellite/satellite_41.88584553507432_-87.67181147737129.png`

**Additional images (require custom trajectory)**:
- ⚠️ Other images in `satellite/` folders - you need to create trajectories for these using `make_trajectory.py`

## Usage

### Option 1: Use Images with Pre-generated Trajectories

Run the demo scripts which use the images that already have trajectories:

```bash
bash demos/inference_demo/demo_cvact.sh 0
bash demos/inference_demo/demo_cvusa.sh 0
bash demos/inference_demo/demo_vigor.sh 0
```

If the required trajectory is missing, the demo scripts will automatically launch `make_trajectory.py` and save the result to:
`work_dirs/visualize_result/{satellite_image_name}/pixels.csv`.

### Option 2: Create Trajectory for Other Images

For other satellite images in the demo data:

```bash
# Example: Create trajectory for a different VIGOR satellite image
python make_trajectory.py \
    --input_img_path demo_data/VIGOR/satellite/satellite_41.85205407245264_-87.6312772533304.png \
    --work_dir work_dirs/visualize_result/

# This creates: work_dirs/visualize_result/satellite_41.85205407245264_-87.6312772533304/pixels.csv

# Then run inference
python inference.py \
    --model checkpoints/s2d_vigor_combine05/checkpoint-437500.pth \
    --sat_img_path demo_data/VIGOR/satellite/satellite_41.85205407245264_-87.6312772533304.png \
    --sky_path demo_data/VIGOR/panorama/ANY_SKY.jpg \
    --position_path work_dirs/visualize_result/satellite_41.85205407245264_-87.6312772533304/pixels.csv \
    --save_video True
```

### Mix Sky Conditions

While trajectories are bound to satellite images, you can **freely mix sky conditions**:

```bash
# Use satellite image A with trajectory A, but sky from image B
python inference.py \
    --model checkpoints/s2d_vigor_combine05/checkpoint-437500.pth \
    --sat_img_path demo_data/VIGOR/satellite/satellite_41.88584553507432_-87.67181147737129.png \
    # Different sky
    --sky_path demo_data/VIGOR/panorama/FfG3-DTW06k2WYpeRjYCUQ,41.852072,-87.631096,.jpg \
    --position_path demo_data/VIGOR/pixels_satellite_41.88584553507432_-87.67181147737129.csv \
    --save_video True
```

## File Naming Convention

### Trajectory Files
Trajectory files are named after their corresponding satellite images:
- Format: `pixels_{satellite_image_name}.csv`
- Example: `satellite_41.88584553507432_-87.67181147737129.png` → `pixels_satellite_41.88584553507432_-87.67181147737129.csv`

### Panorama and Sky Masks
Panorama and sky mask files are matched by image ID but can be used with any satellite image for sky control.

## What You Can Mix

✅ **CAN mix**: Satellite image + ANY panorama/sky  
❌ **CANNOT mix**: Trajectory from one satellite with a different satellite  

## Creating New Trajectories

### Interactive Mode (Requires Display)

```bash
python make_trajectory.py \
    --input_img_path demo_data/VIGOR/satellite/YOUR_IMAGE.png \
    --work_dir work_dirs/visualize_result/
```

**Steps:**
1. A window will open showing the satellite image
2. Click and drag to draw your desired driving path
3. Release mouse button when done
4. Close the window - trajectory is automatically saved

### Requirements
- Graphical display (X11 or local screen)
- For remote servers: use `ssh -X` for X11 forwarding
- Or run locally and copy the generated CSV to the server

## Tips

1. **Start with pre-made trajectories**: Use the 3 ready-to-run demos first
2. **Experiment with sky**: Try different panorama images as sky source
3. **Create custom trajectories**: Use `make_trajectory.py` for other satellite images
4. **Reuse sky conditions**: The 4 VIGOR panoramas provide diverse lighting/weather conditions

## Demo Results (Optional)

We include a small set of example outputs for quick preview:
- `demo_results/vigor/mesh.obj`: Color mesh visualization example
- `demo_results/vigor/vid.gif`: Video preview

## Summary

- **3 satellite images** have pre-generated trajectories (ready to run)
- **7 additional satellite images** for testing (need trajectory creation)
- **All panorama images** can be used as sky source with any satellite
- **Trajectories are specific** to their satellite image - don't mix!
