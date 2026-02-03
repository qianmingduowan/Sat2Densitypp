#!/bin/bash
# Demo script for VIGOR dataset inference

GPU_DEVICE=${1:-0}

# Demo paths
DEMO_DIR="demo_data/VIGOR"
WORK_DIR="work_dirs/visualize_result"
SAT_IMG="${DEMO_DIR}/satellite/satellite_37.74452981131594_-122.42897714405287.png"
SKY_IMG="${DEMO_DIR}/panorama/1TUzL9qFaLoPLBDOEDz1YA,41.854603,-87.631163,.jpg"


# Output settings
SAVE_VIDEO=True
SAVE_SKY=True
SAVE_STREET=True
SAVE_SAT=True
SAVE_SHAPE=True
SEED=0

echo "=== Sat2Density++ VIGOR Demo ==="
echo "Satellite image: $SAT_IMG"
echo "Sky image: $SKY_IMG"
SAT_BASENAME="$(basename "$SAT_IMG")"
SAT_STEM="${SAT_BASENAME%.*}"
DEMO_TRAJECTORY="${DEMO_DIR}/pixels_${SAT_STEM}.csv"
AUTO_TRAJECTORY="${WORK_DIR}/${SAT_STEM}/pixels.csv"
echo "Trajectory: auto (prefer demo csv, otherwise generate to $AUTO_TRAJECTORY)"
echo "GPU: $GPU_DEVICE"
echo ""
echo "Available satellite images in $DEMO_DIR/satellite/:"
for img in $DEMO_DIR/satellite/*.png; do
    basename="$(basename "$img")"
    stem="${basename%.png}"
    demo_csv="$DEMO_DIR/pixels_${stem}.csv"
    auto_csv="$WORK_DIR/${stem}/pixels.csv"
    if [ -f "$demo_csv" ]; then
        echo "  ✅ $basename (has demo trajectory)"
    elif [ -f "$auto_csv" ]; then
        echo "  ✅ $basename (has generated trajectory)"
    else
        echo "  ⚠️  $basename (needs trajectory - will launch make_trajectory.py)"
    fi
done
echo ""
echo "Available sky images in $DEMO_DIR/panorama/:"
ls -1 $DEMO_DIR/panorama/
echo ""

# Check if demo data exists
if [ ! -f "$SAT_IMG" ]; then
    echo "Error: Demo satellite image not found at $SAT_IMG"
    exit 1
fi

if [ ! -f "$SKY_IMG" ]; then
    echo "Error: Demo sky image not found at $SKY_IMG"
    exit 1
fi

POSITION_PATH=""
if [ -f "$DEMO_TRAJECTORY" ]; then
    POSITION_PATH="$DEMO_TRAJECTORY"
elif [ -f "$AUTO_TRAJECTORY" ]; then
    POSITION_PATH="$AUTO_TRAJECTORY"
else
    echo "Trajectory not found. Launching make_trajectory.py to create one..."
    python make_trajectory.py --input_img_path "$SAT_IMG" --work_dir "$WORK_DIR"
    if [ -f "$AUTO_TRAJECTORY" ]; then
        POSITION_PATH="$AUTO_TRAJECTORY"
    else
        echo "Error: Trajectory file not found at $AUTO_TRAJECTORY"
        exit 1
    fi
fi

# Run inference
echo "Running inference..."
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python inference.py \
    --sat_img_path $SAT_IMG \
    --sky_path $SKY_IMG \
    --position_path $POSITION_PATH \
    --save_video $SAVE_VIDEO \
    --save_sky $SAVE_SKY \
    --save_street $SAVE_STREET \
    --save_sat $SAVE_SAT \
    --save_shape $SAVE_SHAPE \
    --truncation_psi 1 \
    --seed $SEED \
    --infer_mode True

echo ""
echo "Done! Results saved to work_dirs/visualize_result/"
echo ""
echo "=== Tips ==="
echo "1. Try different sky conditions:"
echo "   Change --sky_path to any panorama in $DEMO_DIR/panorama/"
echo ""
echo "2. Use other satellite images (trajectory will auto-generate if missing):"
echo "   a. Change --sat_img_path to another image in $DEMO_DIR/satellite/"
echo "   b. If no demo trajectory exists, the script will launch make_trajectory.py"
echo ""
echo "3. Remember: Each trajectory is specific to its satellite image!"
