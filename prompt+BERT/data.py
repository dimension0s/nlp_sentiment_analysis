from torch.utils.data import DataLoader,Dataset
import os
# 1.构建数据集
# 1.1）构建模板
def get_prompt(x):
    prompt=f'总体上来说很[MASK]。{x}'
    return {
        'prompt':prompt,
        'mask_offset':prompt.find('[MASK]')
    }

# 1.2)构建评价指标
def get_verbalizer(tokenizer,vtype):
    assert vtype in ['base','virtual']
    return {
        'pos': {'token': '好', 'id': tokenizer.convert_tokens_to_ids("好")},
        'neg': {'token': '差', 'id': tokenizer.convert_tokens_to_ids("差")}
    } if vtype == 'base' else{
        'pos':
            {'token': '[POS]', 'id': tokenizer.convert_tokens_to_ids("[POS]"),
             'description': '好的、优秀的、正面的评价、积极的态度'},
        'neg':
            {'token': '[NEG]', 'id': tokenizer.convert_tokens_to_ids("[NEG]"),
             'description': '差的、糟糕的、负面的评价、消极的态度'}
    }

# 1.3)加载数据集：使用中文情感分析语料库 ChnSentiCorp 作为数据集
class ChnSentiCrop(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                items = line.strip().split('\t')
                assert len(items) == 2
                prompt_data = get_prompt(items[0])
                Data[idx] = {
                    'comment': items[0],
                    'prompt': prompt_data['prompt'],
                    'mask_offset': prompt_data['mask_offset'],
                    'label': items[1],
                }

        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = ChnSentiCrop("./data/train.txt")
valid_data = ChnSentiCrop("./data/dev.txt")
test_data = ChnSentiCrop("./data/test.txt")
