import swanlab
import torch
import torch.optim as optim
import torch.nn.functional as F

from data import test_loader
from data import train_loader
from model import Net

# 超参数定义
n_epochs = 3
learning_rate = 0.01
momentum = 0.5
log_interval = 10
# 设计随机种子
random_seed = 1
torch.manual_seed(random_seed)

# 初始化网络和优化器
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

# 使用gpu进行训练

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

# 初始化swanlab 因为在train和test函数中进行了测试与参数记录，所以初始化得提前
swanlab.init(
    experiment_name="handwriting_recognition",
    description="handwriting recognition with datasets MNIST.",
    config={
        "model": "cnn",
        "dataset": "MNIST",
        "learning_rate": learning_rate,
    },
    logdir="./logs"
)


def train(epoch):
    # 继承自nn.Module，将模型设置为训练模式
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 清除旧的梯度
        optimizer.zero_grad()
        # 计算当前批次输出
        output = network(data)
        # 利用负对数似然计算损失
        loss = F.nll_loss(output, target)
        # 进行反向传播
        loss.backward()
        # 更新网络参数
        optimizer.step()
        # 每隔一定批次打印训练信息，保存当前模型和优化器的信息
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            swanlab.log({"train_loss": loss.item()})
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


# 训练一个epoch
# train(1)


def test():
    # 将模型设置为评估模式
    network.eval()
    test_loss = 0
    correct = 0
    # 评估模型不需要进行模型参数的更新，于是禁用梯度计算，不会计算反向传播的梯度
    with torch.no_grad():
        for data, target in test_loader:
            # 网络输出
            output = network(data)
            # 网络损失 item()将损失值从tensor变为python数值
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            swanlab.log({"test_loss": test_loss / len(test_loader.dataset)})
            # 从网络输出中找到概率最高的类别索引，即预测结果 [1]表示获取最大值的索引 pred是一个tensor
            pred = output.data.max(1, keepdim=True)[1]
            # view_as()保证维度相同，再判断pred与target是否一直，每一批次累加到correct中
            correct += pred.eq(target.data.view_as(pred)).sum()
            swanlab.log({"test_acc": 100. * correct / len(test_loader.dataset)})
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 开始正确训练之前，可以检测测试集的准确率（应该为10%上下）
# test()


# 正式模型训练
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# 初始化新的网络和优化器
new_network = Net()
new_optimizer = optim.SGD(new_network.parameters(), lr=learning_rate, momentum=momentum)

# 新的变量挂载网络和优化器当前的内部状态
network_state_dict = torch.load('./model.pth')
optimizer_state_dict = torch.load('./optimizer.pth')
new_network.load_state_dict(network_state_dict)
new_optimizer.load_state_dict(optimizer_state_dict)

for i in range(4, 9):
    test_counter.append(i * len(train_loader.dataset))
    train(i)
    test()
