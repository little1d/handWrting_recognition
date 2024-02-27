"""
data_test 查看测试数据的组成
"""
from data import test_loader

# 对test_loader进行迭代 返回包含两个元素的元组：一批图像和相应的标签
examples = enumerate(test_loader)
batch_idx, (examples_data, examples_targets) = next(examples)
# 索引批次 ===> 0
print(batch_idx)
# 张量的形状 torch.Size([1000, 1, 28, 28])
print(examples_data.shape)
# tensor张量，保存图片数据
print(examples_data)
# 标签
print(examples_targets)