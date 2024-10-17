# 5.模型测试
import json
from model import model,to_device
from collate_fn import train_dataloader,valid_dataloader,test_dataloader
from data import test_data
from sklearn.metrics import classification_report
import torch
from tqdm.auto import tqdm

model.load_state_dict(torch.load('epoch_3_valid_macrof1_94.999_microf1_95.000_model_weights.bin'))
with torch.no_grad():
    print('evaluating on test set...')
    true_labels, predictions, probs = [], [], []
    for batch_data in tqdm(test_dataloader):
        true_labels += batch_data['labels']
        batch_data = to_device(batch_data)
        outputs = model(**batch_data)
        pred = outputs[1]
        predictions += pred.argmax(dim=-1).cpu().numpy().tolist()
        probs += torch.nn.functional.softmax(pred, dim=-1)
    save_resluts = []
    for s_idx in tqdm(range(len(test_data))):
        save_resluts.append({
            "comment": test_data[s_idx]['prompt'],
            "label": true_labels[s_idx],
            "pred": predictions[s_idx],
            "prob": {'neg': probs[s_idx][0].item(), 'pos': probs[s_idx][1].item()}
        })
    metrics = classification_report(true_labels, predictions, output_dict=True)
    pos_p, pos_r, pos_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
    neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
    macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
    print(
        f"pos: {pos_p * 100:>0.2f} / {pos_r * 100:>0.2f} / {pos_f1 * 100:>0.2f}, neg: {neg_p * 100:>0.2f} / {neg_r * 100:>0.2f} / {neg_f1 * 100:>0.2f}")
    print(f"Macro-F1: {macro_f1 * 100:>0.2f} Micro-F1: {micro_f1 * 100:>0.2f}\n")
    print('saving predicted results...')
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for example_result in save_resluts:
            f.write(json.dumps(example_result, ensure_ascii=False) + '\n')



