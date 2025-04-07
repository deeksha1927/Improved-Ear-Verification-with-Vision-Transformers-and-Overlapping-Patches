#!/bin/bash
#$ -M darun@nd.edu      # Email address for job notification
#$ -m ae               # Send mail when job begins, ends and aborts
#$ -q gpu
#$ -l gpu_card=2
#$ -pe smp 24
#$ -N vit_s_dp005_mask_0_uerc-awe_2404_247655_100_112x112_p32_s32  # Specify job name
#$ -o /afs/crc.nd.edu/user/d/darun/insightface/recognition/arcface_torch/logs/    # stdout



MODEL_NAME="vit_s_dp005_mask_0_uerc-awe_2404_247655_100_112x112_p32_s32"
SOURCE_DIR="/store01/flynn/darun"
DESTINATION_DIR="${SOURCE_DIR}/features_ear"
SAVE_DIR="plots_ear"
CONFIG_FILE="configs/wf42m_pfc03_40epoch_64gpu_vit_s"

# Load modules and activate the conda environment
module load conda
source activate insightface


# Run inference script

DATASETS=("awe" "awe_lr" "awe_lr_rotated_cropped" "awe_rotated_crop" "DIAST_lr" "opib_lr" "opib_lr_rotated_crop" "AWE-Ex_New_images_rotated_crop" "EarVN1.0_rotated_crop" "AWE-Ex_New_images_rotated_crop_lr" "EarVN1.0_rotated_crop_lr") 

for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"

    # Run the inference script for the current dataset
    python inf_batch.py --network 'vit_s_dp005_mask_0' \
                        --model-name "$MODEL_NAME" \
                        --dataset "$DATASET" \
                        --batch-size 128 \
                        --source "$SOURCE_DIR" \
                        --destination "$DESTINATION_DIR" \
                        --patch_size 32 \
                        --stride 32    

done 
# Deactivate the current environment and activate the next environment
conda deactivate
source activate arcface_extract

for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"

    # Run plot generation script for the current dataset
    python plot_scores_noloop.py --model-name "$MODEL_NAME" \
                                 --dataset "$DATASET" \
                                 --home "$SOURCE_DIR" \
                                 --save-dir "$SAVE_DIR" 

    
done

ROOT="${SOURCE_DIR}/${SAVE_DIR}/${MODEL_NAME}"
OUT="${ROOT}/${MODEL_NAME}.txt"
python save_dprimes.py "$ROOT" "$OUT"

