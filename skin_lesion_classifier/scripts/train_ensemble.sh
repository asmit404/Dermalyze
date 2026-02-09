#!/bin/bash

################################################################################
# Train an Ensemble of 3 Models for Skin Lesion Classification
################################################################################
# This script trains 3 models with different random seeds to create a diverse
# ensemble. Each model is trained independently with the same architecture
# but different initialization and data shuffling.
#
# Usage:
#   bash scripts/train_ensemble.sh [base_output_dir]
#
# Example:
#   bash scripts/train_ensemble.sh outputs/ensemble_run1
################################################################################

set -e  # Exit on error

# Configuration
BASE_OUTPUT_DIR="${1:-outputs/ensemble_$(date +%Y%m%d_%H%M%S)}"
SEEDS=(7 13 21)  # Three different random seeds for diversity
MODEL_NAMES=("model_1" "model_2" "model_3")
CONFIG_FILE="config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}Training Ensemble of 3 Models${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""
echo -e "Base output directory: ${GREEN}${BASE_OUTPUT_DIR}${NC}"
echo -e "Random seeds: ${YELLOW}${SEEDS[@]}${NC}"
echo ""

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Save ensemble metadata
METADATA_FILE="${BASE_OUTPUT_DIR}/ensemble_metadata.json"
cat > "${METADATA_FILE}" << EOF
{
  "ensemble_name": "$(basename ${BASE_OUTPUT_DIR})",
  "created_at": "$(date -Iseconds)",
  "num_models": 3,
  "seeds": [${SEEDS[0]}, ${SEEDS[1]}, ${SEEDS[2]}],
  "models": [
    {
      "name": "${MODEL_NAMES[0]}",
      "seed": ${SEEDS[0]},
      "output_dir": "${BASE_OUTPUT_DIR}/${MODEL_NAMES[0]}"
    },
    {
      "name": "${MODEL_NAMES[1]}",
      "seed": ${SEEDS[1]},
      "output_dir": "${BASE_OUTPUT_DIR}/${MODEL_NAMES[1]}"
    },
    {
      "name": "${MODEL_NAMES[2]}",
      "seed": ${SEEDS[2]},
      "output_dir": "${BASE_OUTPUT_DIR}/${MODEL_NAMES[2]}"
    }
  ]
}
EOF

echo -e "${GREEN}✓${NC} Created ensemble metadata: ${METADATA_FILE}"
echo ""

# Function to create temporary config with modified seed
create_temp_config() {
    local seed=$1
    local temp_config=$2
    
    # Copy config and modify seed
    sed "s/seed: [0-9]*/seed: ${seed}/g" "${CONFIG_FILE}" > "${temp_config}"
}

# Train each model
for i in {0..2}; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    SEED="${SEEDS[$i]}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}"
    TEMP_CONFIG="/tmp/config_ensemble_${MODEL_NAME}_${SEED}.yaml"
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Training Model ${i+1}/3: ${MODEL_NAME}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Seed: ${YELLOW}${SEED}${NC}"
    echo -e "Output: ${GREEN}${OUTPUT_DIR}${NC}"
    echo ""
    
    # Create temporary config with modified seed
    create_temp_config "${SEED}" "${TEMP_CONFIG}"
    
    # Train the model
    echo -e "${YELLOW}Starting training...${NC}"
    python src/train.py \
        --config "${TEMP_CONFIG}" \
        --output "${OUTPUT_DIR}"
    
    # Clean up temp config
    rm -f "${TEMP_CONFIG}"
    
    # Check if training was successful
    if [ -f "${OUTPUT_DIR}/checkpoint_best.pt" ]; then
        echo -e "${GREEN}✓${NC} Model ${MODEL_NAME} trained successfully!"
        echo -e "   Best checkpoint: ${OUTPUT_DIR}/checkpoint_best.pt"
    else
        echo -e "${RED}✗${NC} Model ${MODEL_NAME} training failed!"
        exit 1
    fi
    
    echo ""
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Ensemble Training Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}All 3 models trained successfully!${NC}"
echo ""
echo -e "Ensemble directory: ${GREEN}${BASE_OUTPUT_DIR}${NC}"
echo ""
echo -e "Trained models:"
for i in {0..2}; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    SEED="${SEEDS[$i]}"
    CHECKPOINT="${BASE_OUTPUT_DIR}/${MODEL_NAME}/checkpoint_best.pt"
    echo -e "  ${i+1}. ${MODEL_NAME} (seed=${SEED})"
    echo -e "     ${CHECKPOINT}"
done
echo ""

# Create convenience script for evaluation
EVAL_SCRIPT="${BASE_OUTPUT_DIR}/evaluate_ensemble.sh"
cat > "${EVAL_SCRIPT}" << EOF
#!/bin/bash
# Automatically generated ensemble evaluation script

python src/evaluate.py \\
    --checkpoint \\
        "${BASE_OUTPUT_DIR}/${MODEL_NAMES[0]}/checkpoint_best.pt" \\
        "${BASE_OUTPUT_DIR}/${MODEL_NAMES[1]}/checkpoint_best.pt" \\
        "${BASE_OUTPUT_DIR}/${MODEL_NAMES[2]}/checkpoint_best.pt" \\
    --test-csv "${BASE_OUTPUT_DIR}/${MODEL_NAMES[0]}/test_split.csv" \\
    --images-dir data/HAM10000/images \\
    --output "${BASE_OUTPUT_DIR}/evaluation_results" \\
    "\$@"
EOF
chmod +x "${EVAL_SCRIPT}"

echo -e "${GREEN}✓${NC} Created evaluation script: ${EVAL_SCRIPT}"
echo ""
echo -e "${YELLOW}To evaluate the ensemble:${NC}"
echo -e "  bash ${EVAL_SCRIPT}"
echo ""
echo -e "${YELLOW}To evaluate with TTA:${NC}"
echo -e "  bash ${EVAL_SCRIPT} --use-tta --tta-mode medium"
echo ""

# Create convenience script for inference
INFERENCE_SCRIPT="${BASE_OUTPUT_DIR}/predict_ensemble.sh"
cat > "${INFERENCE_SCRIPT}" << 'EOF'
#!/bin/bash
# Automatically generated ensemble inference script

if [ -z "$1" ]; then
    echo "Usage: $0 <image_path>"
    exit 1
fi

IMAGE_PATH="$1"

python src/inference.py \
    --ensemble \
    --checkpoint \
        "BASE_OUTPUT_DIR/MODEL_1/checkpoint_best.pt" \
        "BASE_OUTPUT_DIR/MODEL_2/checkpoint_best.pt" \
        "BASE_OUTPUT_DIR/MODEL_3/checkpoint_best.pt" \
    --image "${IMAGE_PATH}"
EOF

# Replace placeholders
sed -i.bak "s|BASE_OUTPUT_DIR|${BASE_OUTPUT_DIR}|g" "${INFERENCE_SCRIPT}"
sed -i.bak "s|MODEL_1|${MODEL_NAMES[0]}|g" "${INFERENCE_SCRIPT}"
sed -i.bak "s|MODEL_2|${MODEL_NAMES[1]}|g" "${INFERENCE_SCRIPT}"
sed -i.bak "s|MODEL_3|${MODEL_NAMES[2]}|g" "${INFERENCE_SCRIPT}"
rm -f "${INFERENCE_SCRIPT}.bak"
chmod +x "${INFERENCE_SCRIPT}"

echo -e "${GREEN}✓${NC} Created inference script: ${INFERENCE_SCRIPT}"
echo ""
echo -e "${YELLOW}To predict on an image:${NC}"
echo -e "  bash ${INFERENCE_SCRIPT} path/to/image.jpg"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Ensemble Ready!${NC}"
echo -e "${GREEN}========================================${NC}"
