python cli.py \
    --trainer.gpus=0         \
    --trainer.max_epochs=40 \
    --trainer.default_root_dir='logs/vanila_mlp' \
    --trainer.log_every_n_steps=5 \
    --model.model_name="vanila" \
    --model.optimizer="adam" \
    --model.learning_rate=0.001 \
    --model.num_classes=5 \
    --data.data_dir="/home/ubuntu/splitted_flowers" \
    --data.batch_size=32 \
    --wandb_project_name="Flower Classification" \
    --wandb_task_name="Vanlina MLP" \
    --use_wandb
    # --lr_scheduler="torch.optim.lr_scheduler.CosineAnnealingLR"