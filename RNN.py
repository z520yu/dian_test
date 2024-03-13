import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def compute_metrics(preds, labels, num_classes):
    # 初始化混淆矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for p, l in zip(preds, labels):
        confusion_matrix[l, p] += 1

    # 计算每个类的精确率和召回率
    precision = torch.diag(confusion_matrix) / confusion_matrix.sum(0)
    recall = torch.diag(confusion_matrix) / confusion_matrix.sum(1)

    # 计算F1-Score
    f1 = 2 * precision * recall / (precision + recall)

    # 计算总的准确率
    accuracy = torch.diag(confusion_matrix).sum() / confusion_matrix.sum()

    # 因为可能存在分母为0的情况，我们需要处理NaN值
    precision[precision != precision] = 0  # 将NaN替换为0
    recall[recall != recall] = 0
    f1[f1 != f1] = 0

    # 计算平均值
    precision_mean = precision.mean().item()
    recall_mean = recall.mean().item()
    f1_mean = f1.mean().item()

    return accuracy.item(), precision_mean, recall_mean, f1_mean


# RNN模型定义
class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.relu(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


# 数据加载和准备
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.FashionMNIST('F_MNIST_data', download=True, train=True, transform=transform)
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = datasets.FashionMNIST('F_MNIST_data', download=True, train=False, transform=transform)
testloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 实例化模型、损失函数和优化器
input_size = 28
hidden_size = 128
output_size = 10
net = BasicRNN(input_size, hidden_size, output_size)
loss_rnn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.003)

# 训练函数
def train_rnn(model, trainloader, criterion, optimizer, epochs):
    total_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        hidden = model.init_hidden(images.shape[0])
        images = images.squeeze().view(-1, 28, 28)  # (batch_size, seq_len, input_size)
        for i in range(images.shape[1]):
            output, hidden = net(images[:, i, :], hidden)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(trainloader)
    print(f' Loss: {total_loss:.4f}')
    return total_loss




# 测试函数
def test_rnn(model, testloader,criterion):
    accuracy_all, precision_all, recall_all, f1_all, total_loss = 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            hidden = model.init_hidden(images.shape[0])
            images = images.squeeze().view(-1, 28, 28)
            for i in range(images.shape[1]):
                output, hidden = model(images[:, i, :], hidden)
            loss = criterion(output, labels).item()
            output=torch.argmax(output,dim=1)
            total_loss += loss
            accuracy, precision_mean, recall_mean, f1_mean = compute_metrics(output, labels, num_classes=10)
            accuracy_all += accuracy
            precision_all += precision_mean
            recall_all += recall_mean
            f1_all += f1_mean
        accuracy_all/= len(testloader)
        precision_all/= len(testloader)
        recall_all/= len(testloader)
        f1_all/= len(testloader)
        loss=total_loss/len(testloader)
        print(f'Accuracy: {accuracy_all:.4f}, Precision: {precision_all:.4f}, Recall: {recall_all:.4f}, F1: {f1_all:.4f}')
        return loss, accuracy_all, precision_all, recall_all, f1_all

train_losses = []
test_losses = []
accuracy_all = []
precision_all = []
recall_all = []
f1_all = []

# 训练模型
epochs = 100


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss=train_rnn(net, trainloader, loss_rnn, optimizer, epochs)
    test_loss,accuracy,precision,recall,f1=test_rnn(net, testloader,loss_rnn)
    # 收集每个epoch的训练损失和测试损失
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    accuracy_all.append(accuracy)
    precision_all.append(precision)
    recall_all.append(recall)
    f1_all.append(f1)

# 绘出指标变化图
import matplotlib.pyplot as plt
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 6))

plt.plot(epochs, accuracy_all, label='Accuracy', marker='o', linestyle='-')
plt.plot(epochs, precision_all, label='Precision', marker='s', linestyle='--')
plt.plot(epochs, recall_all, label='Recall', marker='^', linestyle='-.')
plt.plot(epochs, f1_all, label='F1 Score', marker='*', linestyle=':')

plt.title('Model Performance Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.grid(True)
plt.show()


# 绘出损失变化图
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train')
plt.plot(epochs, test_losses, label='Test')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print("Done!")
