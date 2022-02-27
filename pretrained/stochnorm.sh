python main.py \
    --mode="stochnorm" \
    --data_dir="./data/ll_classes/" \
    --num_classes=30 \
    --image_size=224 \
    --batch_size=128 \
    --lr=3e-5 \
    --gpus \
    --num_workers=4 \
    --num_epochs=100 \
    --val_every_n_epochs=1 \
    --seed=42 \
    --use_wandb \
    --project_name "AIVN Pretrained" \
    --run_name="little_little stochnorm"

python main.py \
    --mode="stochnorm" \
    --data_dir="./data/lvl_classes/" \
    --num_classes=30 \
    --image_size=224 \
    --batch_size=128 \
    --lr=3e-5 \
    --gpus \
    --num_workers=4 \
    --num_epochs=100 \
    --val_every_n_epochs=1 \
    --seed=42 \
    --use_wandb \
    --project_name "AIVN Pretrained" \
    --run_name="little_very_little stochnorm"

python main.py \
    --mode="stochnorm" \
    --data_dir="./data/ml_classes/" \
    --num_classes=400 \
    --image_size=224 \
    --batch_size=128 \
    --lr=3e-5 \
    --gpus \
    --num_workers=4 \
    --num_epochs=100 \
    --val_every_n_epochs=1 \
    --seed=42 \
    --use_wandb \
    --project_name "AIVN Pretrained" \
    --run_name="many_little stochnorm"

python main.py \
    --mode="stochnorm" \
    --data_dir="./data/mvl_classes/" \
    --num_classes=400 \
    --image_size=224 \
    --batch_size=128 \
    --lr=3e-5 \
    --gpus \
    --num_workers=4 \
    --num_epochs=100 \
    --val_every_n_epochs=1 \
    --seed=42 \
    --use_wandb \
    --project_name "AIVN Pretrained" \
    --run_name="many_very_little stochnorm"