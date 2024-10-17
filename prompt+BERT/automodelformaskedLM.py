# 2.构建模型
# 2.1）为了做对比，首先尝试在不微调的情况下直接预测情感极性，只摘取第一个样本

import torch
from transformers import AutoModelForMaskedLM
from collate_fn import tokenizer

checkpoint = 'bert-base-chinese'
model = AutoModelForMaskedLM.from_pretrained(checkpoint)

text = "总体上来说很[MASK]。这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。"
inputs = tokenizer(text, return_tensors='pt')
token_logits = model(**inputs).logits

# 会返回行索引和列索引构成的元组：（行索引，列索引），[1]:取列索引
# 初始形状：[batch_size,seq_len],取完mask_token_id后的形状：关于列索引的二维张量,如下：
mask_token_index = torch.where(inputs['input_ids']==tokenizer.mask_token_id)[1]

# logits:返回（batch_size,seq_len,vocab_size）:表示每个位置上每个词汇的logits,
# 取第一个样本，返回二维张量：[[MASK]列索引数,vocab_size]
# 即返回每个[MASK]在所有词汇上的分数logits
mask_token_logits = token_logits[0,mask_token_index,:]

# 基于词汇表的维度，取最高的前5个[MASK]的logits值，
# indices[0]:选择第一个[MASK]位置上的top5词汇，形状：(5,)
# tolist():返回python列表
top_5_tokens = torch.topk(mask_token_logits,5,dim=1).indices[0].tolist()

for token in top_5_tokens:
    # 将token替换为[MASK]候选词，循环遍历
    print(f"'>>>{text.replace(tokenizer.mask_token,tokenizer.decode([token]))}'")