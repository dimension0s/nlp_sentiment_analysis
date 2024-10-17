# 3.模型训练和验证，测试函数
# 3.1）训练函数

from tqdm.auto import tqdm
import os, random
from model import to_device

def train_loop(dataloader,model,optimizer,lr_scheduler,epoch,total_loss):
    total_loss = 0.
    total = 0

    model.train()

    progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step,batch_data in progress_bar:
        batch_data = to_device(batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss/(step+1)
        progress_bar.set_description(f'epoch:{epoch},loss:{avg_loss:.4f}')
        progress_bar.update(1)
    return total_loss





