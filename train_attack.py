from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import random
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from src.util import data_split
from src.attack_utils import create_attack
from src.model_zoo import ShadowModel, SimpleNet, MobileNet
from src.gtsrb import GTSRB


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, required=True, help='cifar10, svhn, gtsrb')
parser.add_argument('--n_model', type=int, default=50, help='Number of clean models')
args = parser.parse_args()

random.seed(datetime.now())

if not os.path.isdir('{}_attack'.format(args.dataset)):
    os.mkdir('{}_attack'.format(args.dataset))
if not os.path.isdir('{}_trigger'.format(args.dataset)):
    os.mkdir('{}_trigger'.format(args.dataset))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load in dataset
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    # First half of the CIFAR-10 test set is used for model training, the second half is for evaluation
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    trainset = data_split(args.dataset, trainset, 'defense', ratio=0.6)
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
    # First half of the CIFAR-10 test set is used for model training, the second half is for evaluation
    trainset = GTSRB(root='./data/gtsrb', split='test', download=True, transform=transform_train)
    testset = GTSRB(root='./data/gtsrb', split='train', download=True, transform=transform_test)
    testset = data_split(args.dataset, testset, 'evaluation', ratio=0.5)
elif args.dataset == 'svhn':
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    # First half of the CIFAR-10 test set is used for model training, the second half is for evaluation
    trainset = torchvision.datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform_train)
    testset = torchvision.datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform_test)
    trainset = data_split(args.dataset, trainset, 'defense', ratio=0.6)
    testset = data_split(args.dataset, testset, 'evaluation', ratio=0.2)
else:
    sys.exit('Incorrect dataset name!')

# Generate attack
i = 0
while i < args.n_model:
    trainset_poisoned, attackset, trigger, target_class = create_attack(args.dataset, trainset, testset)

    # Load in the datasets
    if args.dataset == 'cifar10':
        batch_size = 32
    elif args.dataset == 'gtsrb' or 'svhn':
        batch_size = 128
    else:
        sys.exit('Invalid dataset!')
    trainloader = torch.utils.data.DataLoader(trainset_poisoned, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    attackloader = torch.utils.data.DataLoader(attackset, batch_size=100, shuffle=False, num_workers=2)

    # Model
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
    # Attack evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(attackloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    asr = 100. * correct / total

    if acc > 60 and asr > 90:
        torch.save(net.state_dict(), './{}_attack/model_{}.pth'.format(args.dataset, i))
        torch.save(trigger, '{}_trigger/trigger_{}'.format(args.dataset, i))
        torch.save(target_class, '{}_trigger/target_class_{}'.format(args.dataset, i))
        print('Model: %s; acc: %.3f; asr: %.3f' % (i, acc, asr))
        i += 1
    else:
        print('Failed attack: acc: %.3f; asr: %.3f ' % (acc, asr))
