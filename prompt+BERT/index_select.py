import torch

# 在指定维度dim上选择指定索引对应的值
def batched_index_select(input, dim, index):
    # 1.将index扩展成与input兼容的形状，注意是兼容
    # dim的作用：决定在哪个维度进行索引
    for i in range(1, len(input.shape)):# 1,2,......len-1
        if i != dim:
            index = index.unsqueeze(i)
    # 2.形状扩展，重点是扩展，扩展后才能正确找到input中具体的元素位置，然后提取元素值
    expanse = list(input.shape)
    expanse[0] = -1  # 不改变第一个维度
    expanse[dim] = -1  # 不改变第 dim 个维度
    index = index.expand(expanse)  # 按照input.shape的形状扩展
    return torch.gather(input, dim, index)

# 举例演示
# 示例1.创建一个输入张量
input_1 = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
print(f'list(input.shape):{list(input_1.shape)}')  # list(input.shape):[3, 3]
print(f'len(input.shape):{len(input_1.shape)}')  # len(input.shape):2

# 创建一个索引张量
index_1 = torch.tensor([[0,2],
                   [1,0]])
print(index_1.shape)  # torch.Size([2, 2])

# 示例2：
input_2 = torch.tensor([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])
print(f'list(input_2.shape):{list(input_2.shape)}')  # list(input.shape):[2, 2, 3]
print(f'len(input_2.shape):{len(input_2.shape)}')  # len(input.shape):3

index_2 = torch.tensor([
[0, 2],
[1, 0]])

output = batched_index_select(input_2,dim=2,index=index_2)
print(f'output:{output}')  # output:tensor([[[ 1,  3],[ 4,  6]],[[ 8,  7],[11, 10]]])
print(f'input_2.shape:{index_2.shape}')  # input_2.shape:torch.Size([2, 2, 3])
print(f'index_2.shape:{index_2.shape}')  # index_2.shape:torch.Size([2, 2])
