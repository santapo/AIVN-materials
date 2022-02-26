run_name="chest_xray"
data_dir="./data/chest_xray/"
num_classes=30
lr=3e-5

python main.py \
    --use_wandb \
    --project_name "AIVN Pretrained" \
    --run_name="$run_name scratch" \
    --mode="scratch" \
    --data_dir=$data_dir \
    --num_classes=$num_classes \
    --image_size=224 \
    --batch_size=128 \
    --lr=$lr \
    --gpus \
    --num_workers=4 \
    --num_epochs=100 \
    --val_every_n_epochs=1 \
    --seed=42

python main.py \
    --use_wandb \
    --project_name "AIVN Pretrained" \
    --run_name="$run_name finetuning" \
    --mode="finetuning" \
    --data_dir=$data_dir \
    --num_classes=$num_classes \
    --image_size=224 \
    --batch_size=128 \
    --lr=$lr \
    --gpus \
    --num_workers=4 \
    --num_epochs=100 \
    --val_every_n_epochs=1 \
    --seed=42

python main.py \
    --use_wandb \
    --project_name "AIVN Pretrained" \
    --run_name="$run_name transfer" \
    --mode="transfer" \
    --data_dir=$data_dir \
    --num_classes=$num_classes \
    --image_size=224 \
    --batch_size=128 \
    --lr=$lr \
    --gpus \
    --num_workers=4 \
    --num_epochs=100 \
    --val_every_n_epochs=1 \
    --seed=42

python main.py \
    --use_wandb \
    --project_name "AIVN Pretrained" \
    --run_name="$run_name pretrained" \
    --mode="pretrained" \
    --data_dir=$data_dir \
    --num_classes=$num_classes \
    --image_size=224 \
    --batch_size=128 \
    --lr=$lr \
    --gpus \
    --num_workers=4 \
    --num_epochs=100 \
    --val_every_n_epochs=1 \
    --seed=42