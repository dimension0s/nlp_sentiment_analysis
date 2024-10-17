# 2.2) 采用automodelformaskedLM的方式不够灵活，下面基于transformers库预训练模型来手工构建模型：
# 对比原来的代码，以下模型做了简化，方便理解，并且并不影响功能
from transformers import BertPreTrainedModel,BertModel,BertConfig
from collate_fn import tokenizer,get_prompt,get_verbalizer,verbalizer,pos_id,neg_id,vtype,checkpoint
from device import device
import torch.nn as nn
from transformers.activations import ACT2FN
import torch
from index_select import batched_index_select

class BertForPrompt(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.bert = BertModel(config,add_pooling_layer=False)
        # 构建自定义预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size),
            ACT2FN[config.hidden_act] if isinstance(config.hidden_act,str)
            else config.hidden_act,
            nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size,config.vocab_size,bias=False) ) # decoder
        self.cls = self.prediction_head[-1]
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.prediction_head[-1].bias = self.bias
        self.post_init()  # 后处理，和初始化模型权重有关

    def forward(self,batch_inputs,batch_mask_idxs,label_word_id,labels=None):
        bert_output = self.bert(**batch_inputs)
        sequence_output = bert_output.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output,1,batch_mask_idxs.unsqueeze(-1)).squeeze(1)
        pred_scores = self.cls(batch_mask_reps)[:,label_word_id]

        # 计算损失
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred_scores, labels)
        return loss, pred_scores  # 返回损失和预测函数

config = BertConfig.from_pretrained(checkpoint)
model = BertForPrompt.from_pretrained(checkpoint,config=config).to(device)

if vtype == 'virtual':
    model.resize_token_embeddings(len(tokenizer))
    print(f"initialize embeddings of {verbalizer['pos']['token']} and {verbalizer['neg']['token']}")
    with torch.no_grad():
        pos_tokenized = tokenizer(verbalizer['pos']['description'])
        pos_tokenized_ids = tokenizer.convert_tokens_to_ids(pos_tokenized)
        neg_tokenized = tokenizer(verbalizer['neg']['description'])
        neg_tokenized_ids = tokenizer.convert_tokens_to_ids(neg_tokenized)
        new_embedding = model.bert.embeddings.word_embeddings.weight[pos_tokenized_ids].mean(axis=0)
        model.bert.embeddings.word_embeddings.weight[pos_id, :] = new_embedding.clone().detach().requires_grad_(True)
        new_embedding = model.bert.embeddings.word_embeddings.weight[neg_tokenized_ids].mean(axis=0)
        model.bert.embeddings.word_embeddings.weight[neg_id, :] = new_embedding.clone().detach().requires_grad_(True)


def to_device(batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(device) for k_, v_ in v.items()
            }
        elif k == 'label_word_id':
            new_batch_data[k] = v
        else:
            new_batch_data[k] = torch.tensor(v).to(device)
    return new_batch_data
