import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 加载MNIST数据集
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

#计算指标
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

# 创建数据加载器
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),  # 第一层，将28x28的图片展平后输入，输出128个节点
            nn.ReLU(),
            nn.Linear(128, 256),  # 第二层，接收第一层的128个节点，输出256个节点
            nn.ReLU(),
            nn.Linear(256, 10)  # 最后一层，输出10个节点，对应10个类别
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 训练过程
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss

# 测试过程
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss= 0
    accuracy_all, precision_all, recall_all, f1_all = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = torch.argmax(pred, dim=1)
            accuracy, precision_mean, recall_mean, f1_mean = compute_metrics(pred, y, num_classes=10)
            accuracy_all += accuracy
            precision_all += precision_mean
            recall_all += recall_mean
            f1_all += f1_mean
    test_loss /= num_batches
    accuracy_all /= num_batches
    precision_all /= num_batches
    recall_all /= num_batches
    f1_all /= num_batches
    print(f"Test Error: \n Accuracy: {(100*accuracy_all):>0.1f}%, Precision: {(100*precision_all):>0.1f}%, Recall: {(100*recall_all):>0.1f}%, F1: {(100*f1_all):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy_all, precision_all, recall_all, f1_all

# 训练模型
epochs = 100
train_losses = []
test_losses = []
accuracy_all = []
precision_all = []
recall_all = []
f1_all = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss=train(train_dataloader, model, loss_fn, optimizer)
    test_loss,accuracy,precision,recall,f1=test(test_dataloader, model, loss_fn)
    # 将trian_loss和test_loss转换为numpy类型
    train_loss = train_loss.cpu().detach().numpy()
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
