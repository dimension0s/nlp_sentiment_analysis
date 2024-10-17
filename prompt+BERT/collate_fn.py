from transformers import AutoTokenizer,AutoConfig
from torch.utils.data import DataLoader
from data import get_prompt,get_verbalizer,train_data,valid_data,test_data

vtype = 'base'

checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if vtype == 'virtual':  # 将新词加入分词器语料库中
    tokenizer.add_special_tokens({'additional_special_tokens':['[POS]','[NEG]']})

verbalizer = get_verbalizer(tokenizer,vtype=vtype)
pos_id, neg_id = verbalizer['pos']['id'], verbalizer['neg']['id']

# 1.3)分批，分词，编码
def collote_fn(batch_samples):
    batch_sentences, batch_mask_idxs, batch_labels = [], [], []
    for sample in batch_samples:
        batch_sentences.append(sample['prompt'])
        encoding = tokenizer(sample['prompt'], truncation=True)
        mask_idx = encoding.char_to_token(sample['mask_offset'])
        assert mask_idx is not None
        batch_mask_idxs.append(mask_idx)
        batch_labels.append(int(sample['label']))

    batch_inputs = tokenizer(
        batch_sentences,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt")
    label_word_id = [neg_id, pos_id]

    return {
        'batch_inputs': batch_inputs,
        'batch_mask_idxs': batch_mask_idxs,
        'label_word_id': label_word_id,
        'labels': batch_labels
    }

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

