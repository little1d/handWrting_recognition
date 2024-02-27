"""
data_test 查看测试数据的组成
"""
from data import test_loader
import matplotlib.pyplot as plt

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

# 绘制图像
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    # 选择图像第一个通道 颜色映射为灰度 不适用插值
    plt.imshow(examples_data[i][0], cmap='gray', interpolation='none')
    plt.title("True value:{}".format(examples_targets[i]))
    # 隐藏刻度
    plt.xticks([])
    plt.yticks([])
plt.show()
