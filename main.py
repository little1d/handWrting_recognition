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
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


# 训练一个epoch
train(1)


def test():
    # 将模型设置为评估模式
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 开始正确训练之前，检测测试集的准确率（应该为10%上下）
test()
