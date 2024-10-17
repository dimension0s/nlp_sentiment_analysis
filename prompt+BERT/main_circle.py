# 4.主训练循环
import os
from transformers import AdamW,get_scheduler
import torch
import random
import numpy as np
from model import model
from collate_fn import train_dataloader,valid_dataloader
from train import train_loop
from test import test_loop

learning_rate = 1e-5
epoch_num = 6

# 设置随机数，保证每次实验结果相同
def seed_everything(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(42)

optimizer = AdamW(model.parameters(),lr=learning_rate)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader)
)

total_loss = 0.
total = 0
best_f1_score = 0.
for epoch in range(epoch_num):
    print(f'Epoch {epoch+1}/{epoch_num}\n---------------------------------------')
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch + 1, total_loss)
    valid_scores = test_loop(valid_dataloader, model, 'Valid')
    macro_f1, micro_f1 = valid_scores['macro avg']['f1-score'], valid_scores['weighted avg']['f1-score']
    f1_score = (macro_f1 + micro_f1) / 2
    if f1_score > best_f1_score:
        best_f1_score = f1_score
        print('saving new weights...\n')
        torch.save(
            model.state_dict(),
            f'epoch_{epoch + 1}_valid_macrof1_{(macro_f1 * 100):0.3f}_microf1_{(micro_f1 * 100):0.3f}_model_weights.bin')
