# Two-Phase Training

## Phase 1: Warmup (5 epochs with frozen backbone)

python src/train.py --config config_warmup.yaml

## Phase 2: Balanced fine-tuning (40 epochs with unfrozen layers)

python src/train.py --config config_balanced.yaml --resume outputs/run_XXX/checkpoint_best.pt

## Performance Tweaks

### Use GPU acceleration (--device mps)

### Use mixed precision (faster + less memory)

### Run on all CPU cores
export OMP_NUM_THREADS=$(sysctl -n hw.logicalcpu)
python src/train.py --config config_warmup.yaml

### Pre-cache dataset to SSD

python -c "from src.data import DataLoader; DataLoader.preprocess(cache=True)"

### Train with cached data

python src/train.py --config config_warmup.yaml --use_cache