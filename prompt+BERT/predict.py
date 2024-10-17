# 6.预测函数封装
import torch
from data import get_verbalizer,get_prompt,test_data
from collate_fn import verbalizer,tokenizer
from model import to_device,model

def predict(comment, tokenizer, model, verbalizer):
    prompt_data = get_prompt(comment)
    prompt = prompt_data['prompt']
    encoding = tokenizer(prompt, truncation=True)  # 或者命名为tokens
    mask_idx = encoding.char_to_token(prompt_data['mask_offset'])  # 将[MASK]转化为token_id
    assert mask_idx is not None  # [MASK]必须存在
    inputs = tokenizer(
        prompt,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    inputs = {
        'batch_inputs': inputs,
        'batch_mask_idxs': [mask_idx],
        'label_word_id': [verbalizer['neg']['id'], verbalizer['pos']['id']]

    }
    inputs = to_device(inputs)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        # logits形状：[batch_size,len(label_word_id)]
        logits = outputs[1]  # 或者：outputs.logits

        prob = torch.nn.functional.softmax(logits, dim=-1)
    # 取分数最大值索引，（从大到小排列），[0]:获取第一个样本
    # item():转化成整数
    pred = logits.argmax(dim=-1)[0].item()
    # prob[0]:指第一个样本
    prob = prob[0][pred].item()
    return pred, prob

# 注意：以上封装函数获取的pred和prob都是第一个样本，这并不是一般做法。
# 一般来说，可以跟随batch_size批次同步prob和pred结果，即对于：
# pred = logits.argmax(dim=-1)[0].item()
# prob = prob[0][pred].item()
# 以上2行代码中的[0]直接去掉，就可以实现对批次中每个样本的预测处理

model.load_state_dict(torch.load('epoch_3_valid_macrof1_94.999_microf1_95.000_model_weights.bin'))

for i in range(5):
    data = test_data[i]
    pred, prob = predict(data['comment'], tokenizer,model,verbalizer)# 注意参数顺序不要放错
    print(f"{data['comment']}\nlabel: {data['label']}\tpred: {pred}\tprob: {prob}")