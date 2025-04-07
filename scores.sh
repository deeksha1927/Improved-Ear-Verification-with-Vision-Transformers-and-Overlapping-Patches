#!/bin/bash
#$ -M darun@nd.edu      # Email address for job notification
#$ -m ae               # Send mail when job begins, ends, and aborts
#$ -pe smp 16
#$ -t 1-5
#$ -N auc_vit_l_dp005_mask_005_p28_s14_original # Specify job name
#$ -o /afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/logs_b/    # stdout


#MODEL="vit_l_dp005_mask_005_p16_s8"
MODEL="vit_l_dp005_mask_005_p28_s14_original"
MODEL_NAME="${MODEL}_${SGE_TASK_ID}"
MODEL_FOLDER="${MODEL}/${MODEL_NAME}"
OUTPUT_FOLDER="results/${MODEL}"
SOURCE_DIR="/store01/flynn/darun"
DESTINATION_DIR="${SOURCE_DIR}/features_ear/"
SAVE_DIR="plots_ear"
COMPLETION_DIR="/afs/crc.nd.edu/user/d/darun/completion_markers_auc/${MODEL}/"
# Load modules and activate the conda environment
module load conda


# Inference
DATASETS=("awe" "awe_lr" "awe_lr_rotated_cropped" "awe_rotated_crop" "DIAST_lr" "opib_lr" "opib_lr_rotated_crop" "AWE-Ex_New_images_rotated_crop" "EarVN1.0_rotated_crop" "AWE-Ex_New_images_rotated_crop_lr" "EarVN1.0_rotated_crop_lr" "WPUT_rotated_crop_lr" "WPUT_rotated_crop" "EarVN1.0_lr" "AWE-Ex_New_images_lr" "WPUT_lr")
# Create completion marker directory
mkdir -p "$COMPLETION_DIR"


source activate arcface_extract

for DATASET in "${DATASETS[@]}"; do
    python aucroc.py     --model-name "$MODEL_FOLDER" \
                                 --dataset "$DATASET" \
                                 --home "$SOURCE_DIR" \
                                 --save-dir "$SAVE_DIR"
done

# Save d-prime results
ROOT="${SOURCE_DIR}/${SAVE_DIR}/${MODEL_FOLDER}"
OUT="${SOURCE_DIR}/${SAVE_DIR}/${MODEL}/auc_${MODEL_NAME}.txt"
python save_auc.py "$ROOT" "$OUT"



# Write completion marker for this task
echo "Task $SGE_TASK_ID completed" > "$COMPLETION_DIR/task_${SGE_TASK_ID}.done"

# Final step for task 1 only
if [ "$SGE_TASK_ID" -eq 1 ]; then
    echo "Waiting for all tasks to complete..."

    # Check for completion of all tasks
    while true; do
        ALL_COMPLETED=true
        for i in $(seq 1 $SGE_TASK_LAST); do
            if [ ! -f "$COMPLETION_DIR/task_${i}.done" ]; then
                ALL_COMPLETED=false
                break
            fi
        done

        if [ "$ALL_COMPLETED" = true ]; then
            break
        fi

        # Wait for a short time before checking again
        sleep 10
    done


    echo "All tasks completed. Generating final data table..."
    ROOT="${SOURCE_DIR}/${SAVE_DIR}/${MODEL}"
    OUT="${SOURCE_DIR}/${SAVE_DIR}/${MODEL}/${MODEL}_auc.csv"
    python make_auc_table.py "$ROOT" "$OUT"
    echo "Data table generated successfully: $OUT"

       
    rm -r "$COMPLETION_DIR"
fi

