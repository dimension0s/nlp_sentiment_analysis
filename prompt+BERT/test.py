# 3.2) 验证（测试）函数
from sklearn.metrics import classification_report
import torch
from model import to_device
import numpy

def test_loop(dataloader,model,mode='Valid'):
    assert mode in ['Valid','Test']
    true_labels, predictions =[], []
    model.eval()
    with torch.no_grad():
        for batch_data in dataloader:
            true_labels += batch_data['labels']
            batch_data = to_device(batch_data)
            outputs = model(**batch_data)
            pred = outputs[1] # 获取预测值
            # detach():创建了一个与 pred 张量相同数据的新张量，但是与计算图分离，
            # 这样就可以将其转换为 NumPy 数组并添加到 predictions 列表中,
            # 且不会影响到梯度计算。
            predictions += pred.argmax(dim=-1).cpu().numpy().tolist()

    metrics = classification_report(true_labels,predictions,output_dict=True)
    pos_p, pos_r, pos_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
    neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
    macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
    print(
        f"pos: {pos_p * 100:>0.2f} / {pos_r * 100:>0.2f} / {pos_f1 * 100:>0.2f}, neg: {neg_p * 100:>0.2f} / {neg_r * 100:>0.2f} / {neg_f1 * 100:>0.2f}")
    print(f"Macro-F1: {macro_f1 * 100:>0.2f} Micro-F1: {micro_f1 * 100:>0.2f}\n")
    return metrics

# 打印的是精确率P，召回率R，F1值，
# 注意精确率Precision和准确率Accuracy的区别
