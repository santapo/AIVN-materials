python cli.py \
    --trainer.gpus=1 \
    --trainer.max_epochs=200 \
    --trainer.default_root_dir='logs/vanila_mlp' \
    --trainer.log_every_n_steps=5 \
    --model.optimizer="adam" \
    --model.learning_rate=0.0001 \
    --model.num_classes=5 \
    --data.data_dir="/root/splitted_flowers" \
    --data.batch_size=32 \
    --wandb_project_name="Flower Classification" \
    --wandb_task_name="Vanlina MLP" \
    --use_wandb
    # --lr_scheduler="torch.optim.lr_scheduler.CosineAnnealingLR"