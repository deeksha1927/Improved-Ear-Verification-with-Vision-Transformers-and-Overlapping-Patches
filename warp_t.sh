#!/bin/bash
#$ -M darun@nd.edu      # Email address for job notification
#$ -m ae               # Send mail when job begins, ends, and aborts
#$ -q gpu
#$ -l h=qa-a10-*|qa-rtx6k-*|qa-l40s-*
#$ -l gpu_card=2
#$ -pe smp 8
#$ -t 1-5
#$ -N warp_vit_t_dp005_mask_0_p28_s28_original # Specify job name
#$ -o /afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/logs_s    # stdout


MODEL="warp_vit_t_dp005_mask_0_p28_s28_original"
PATCH=28
STRIDE=28
BATCH_SIZE=128


MODEL_NAME="${MODEL}_${SGE_TASK_ID}"
MODEL_FOLDER="${MODEL}/${MODEL_NAME}"
OUTPUT_FOLDER="results/${MODEL}"
SOURCE_DIR="/store01/flynn/darun/rotated_Ears/rotated-ears/"
DESTINATION_DIR="${SOURCE_DIR}/features_ear/"
SAVE_DIR="plots_ear"
CONFIG_FILE="configs/wf42m_pfc03_40epoch_8gpu_vit_t"
COMPLETION_DIR="/afs/crc.nd.edu/user/d/darun/completion_markers/${MODEL}/"
random_number=$((10 + RANDOM % (9999 - 10 + 1)))

lst2=('0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' \
'21' '22' '23' '24' '25' '26' '27' '28' '29' '30' '31' '32' '33' '34' '35' '36' '37' '38' '39' '40' '41' \
'42' '43' '44' '45' '46' '47' '48')
# Create completion marker directory
mkdir -p "$COMPLETION_DIR"

# Load modules and activate the conda environment
module load conda
source activate insightface

# Training
torchrun --nproc_per_node=2 --master_port=2561${lst2[$SGE_TASK_ID-1]} train_v2.py $CONFIG_FILE $MODEL_NAME $random_number $PATCH $STRIDE $OUTPUT_FOLDER $BATCH_SIZE
# Inference
DATASETS=("WPUT_warped_lr" "WPUT_warped" "opib_warped_lr" "opib_warped" "earvn_warped_lr" "EarVN1.0_warped" "awe_warped_lr" "awe_warped" "AWE-Ex_New_images_warped_lr" "AWE-Ex_New_images_warped" )

for DATASET in "${DATASETS[@]}"; do
    python inf_batch.py --network "vit_t_dp005_mask0" \
                        --model-name "$MODEL_FOLDER" \
                        --dataset "$DATASET" \
                        --batch-size "$BATCH_SIZE" \
                        --source "$SOURCE_DIR" \
                        --destination "$DESTINATION_DIR" \
                        --patch_size "$PATCH" \
                        --stride "$STRIDE"
done

# Plot generation
conda deactivate
source activate arcface_extract
for DATASET in "${DATASETS[@]}"; do
    python plot_scores_noloop.py --model-name "$MODEL_FOLDER" \
                                 --dataset "$DATASET" \
                                 --home "$SOURCE_DIR" \
                                 --save-dir "$SAVE_DIR"
done



# Save d-prime results
ROOT="${SOURCE_DIR}/${SAVE_DIR}/${MODEL_FOLDER}"
OUT="${SOURCE_DIR}/${SAVE_DIR}/${MODEL}/dprime_${MODEL_NAME}_seed$random_number.txt"
python save_dprimes.py "$ROOT" "$OUT"


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
    OUT="${SOURCE_DIR}/${SAVE_DIR}/${MODEL}/${MODEL}_dprime.csv"
    python make_d_table_warp.py "$ROOT" "$OUT"
    echo "Data table generated successfully: $OUT"



    # Clean up
    #rm -r "/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/results/${MODEL}"
    #rm -r "$COMPLETION_DIR"
fi




