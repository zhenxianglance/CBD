from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from src.util import data_split
from src.model_zoo import ShadowModel, SimpleNet, MobileNet
from src.gtsrb import GTSRB


parser = argparse.ArgumentParser(description='Train clean shadow models for null modeling')
parser.add_argument('--dataset', type=str, required=True, help='cifar10, svhn, gtsrb')
parser.add_argument('--n_model', type=int, default=100, help='Number of shadow models')
args = parser.parse_args()

if not os.path.isdir('{}_shadow'.format(args.dataset)):
    os.mkdir('{}_shadow'.format(args.dataset))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load in dataset
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    trainset = data_split(args.dataset, trainset, 'defense', ratio=0.9)  # 5000 samples
    testset = data_split(args.dataset, testset, 'evaluation', ratio=0.5)
elif args.dataset == 'gtsrb':
    transform_train = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    trainset = GTSRB(root='./data/gtsrb', split='test', download=True, transform=transform_train)
    testset = GTSRB(root='./data/gtsrb', split='train', download=True, transform=transform_test)
    trainset = data_split(args.dataset, trainset, 'defense', ratio=0.605)  # 5000 samples
    testset = data_split(args.dataset, testset, 'evaluation', ratio=0.8)
elif args.dataset == 'svhn':
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform_train)
    testset = torchvision.datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform_test)
    trainset = data_split(args.dataset, trainset, 'defense', ratio=0.808)  # 5000 samples
    testset = data_split(args.dataset, testset, 'evaluation', ratio=0.2)
else:
    sys.exit('Incorrect dataset name!')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# Train shadow models
for i in range(args.n_model):
    # Model architecture
    if args.dataset == 'cifar10':
        net = ShadowModel()
    elif args.dataset == 'gtsrb':
        net = SimpleNet()
    elif args.dataset == 'svhn':
        net = MobileNet(num_classes=10)
    else:
        sys.exit('Incorrect dataset name!')
    net = net.to(device)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Training
    for epoch in range(50):
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    # Evaluation
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print('Model: %s; acc: %.3f' % (i, acc))

    # Save model
    torch.save(net.state_dict(), './{}_shadow/model_{}.pth'.format(args.dataset, i))

