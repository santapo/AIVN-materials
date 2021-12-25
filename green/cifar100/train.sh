python cli.py fit \
    --trainer.gpus=1 \
    --trainer.max_epochs=200 \
    --model.optimizer="sgd" \
    --model.learning_rate=0.1 \
    --data.batch_size=128
    # --lr_scheduler="torch.optim.lr_scheduler.CosineAnnealingLR"