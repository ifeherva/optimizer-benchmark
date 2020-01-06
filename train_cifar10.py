import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models.resnet_cifar10 import ResNet34
from optimizers import parse_optimizer, supported_optimizers


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='resnet34', type=str, help='model',
                        choices=['resnet', 'densenet'])
    parser.add_argument('--optim', type=str, help='optimizer', required=True,
                        choices=supported_optimizers())
    parser.add_argument('--seed', type=int, default=123, help='Random seed to use. default=123.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--wandb-project', type=str, help='Project name on Weights&Biases')

    args, optim_args = parser.parse_known_args()
    return args, optim_args


def build_dataset():
    """Build CIFAR10 train and test data loaders. Will download datasets if needed."""
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True,
                              num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


def build_model(model, device):
    net = {
        'resnet34': ResNet34,
    }[model]()
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        cudnn.deterministict = True

    return net


def test(net, device, data_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('Test acc %.3f' % accuracy)

    return accuracy


def train_epoch(net, epoch, device, data_loader, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), desc='Epoch {}'.format(epoch)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)
    print('train loss %.6f' % train_loss)

    return accuracy, train_loss


def train_cifar10(opt, optimizer_opts):
    # Set random seed
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    train_loader, test_loader = build_dataset()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = build_model(opt.model, device)

    criterion = CrossEntropyLoss()
    optimizer, optimizer_run_name = parse_optimizer(opt.optim, optimizer_opts, net.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    run_name = 'cifar10_{}'.format(
        optimizer_run_name
    )

    train_accuracies = []
    test_accuracies = []
    train_loss_data = []
    best_acc = 0

    if opt.wandb_project:
        import wandb
        wandb.init(project=opt.wandb_project, name=run_name)

    for epoch in range(opt.epochs):
        train_acc, train_loss = train_epoch(net, epoch, device, train_loader, optimizer, criterion)
        test_acc = test(net, device, test_loader)

        scheduler.step()

        if opt.wandb_project:
            wandb.log({
                'Training Loss': train_loss,
                'Training Accuracy': train_acc,
                'Test Accuracy': test_acc
            })

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_loss_data.append(train_loss)


if __name__ == '__main__':
    train_cifar10(*parse_args())
