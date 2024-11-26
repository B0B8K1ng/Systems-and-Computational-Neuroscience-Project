import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn = layer.BatchNorm2d(out_channels)
        self.neuron = neuron.IFNode(surrogate_function=surrogate.ATan())
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.neuron(x)
        return x

# 定义SNN模型
class CSNN(nn.Module):
    def __init__(self, T=4):
        super().__init__()
        self.T = T  # 仿真时长

        self.conv_layers = nn.Sequential(
            ConvBlock(3, 128),
            ConvBlock(128, 128),
            layer.MaxPool2d(2, 2),
            
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            layer.MaxPool2d(2, 2),
            
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            layer.MaxPool2d(2, 2)
        )
        
        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(512 * 4 * 4, 1024),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(1024, 10)
        )

    def forward(self, x):
        # 静态图像转换为脉冲序列
        out_spikes = 0
        for t in range(self.T):
            x_seq = x.clone()
            
            # 通过卷积层
            x_seq = self.conv_layers(x_seq)
            
            # 通过全连接层
            x_seq = self.fc(x_seq)
            
            out_spikes += x_seq

        return out_spikes / self.T

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 重置神经元状态
        functional.reset_net(model)
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_description(f'Epoch {epoch} Train: Loss {train_loss/(batch_idx+1):.4f} Acc {100.*correct/total:.2f}%')

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Test'):
            data, target = data.to(device), target.to(device)
            
            # 重置神经元状态
            functional.reset_net(model)
            
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

# 主训练流程
def main():
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = CSNN(T=4).to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 训练循环
    best_acc = 0
    for epoch in range(200):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        scheduler.step()
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'snn_cifar10_best.pth')
            print(f'Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()