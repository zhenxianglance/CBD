from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import random
from datetime import datetime

from src.util import data_split, smooth, get_freq
from src.model_zoo import ShadowModel, SimpleNet, MobileNet
from src.gtsrb import GTSRB


parser = argparse.ArgumentParser(description='Obtain the LDP statistics')
parser.add_argument('--dataset', type=str, required=True, help='cifar10, svhn, gtsrb')
parser.add_argument('--sigma', type=float, default=2.0, help='Standard deviation of the Gaussian noise')
parser.add_argument('--stat_type', type=str, required=True, help='shadow, clean, attack')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(datetime.now())

# Load in dataset
if args.dataset == 'cifar10':
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    testset = data_split(args.dataset, testset, 'evaluation', ratio=0.5)
elif args.dataset == 'gtsrb':
    transform_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    testset = GTSRB(root='./data/gtsrb', split='train', download=True, transform=transform_test)
    testset = data_split(args.dataset, testset, 'evaluation', ratio=0.5)
elif args.dataset == 'svhn':
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform_test)
    testset = data_split(args.dataset, testset, 'evaluation', ratio=0.5)
else:
    sys.exit('Incorrect dataset name!')
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Compute LDP
probs_max = []
for i in range(len(os.listdir('{}_{}'.format(args.dataset, args.stat_type)))):

    # Prepare model
    if args.dataset == 'cifar10':
        net = ShadowModel()
    elif args.dataset == 'gtsrb':
        net = SimpleNet()
    elif args.dataset == 'svhn':
        net = MobileNet(num_classes=10)
    else:
        sys.exit('Incorrect dataset name!')
    net = net.to(device)
    net.load_state_dict(torch.load('./{}_{}/model_{}.pth'.format(args.dataset, args.stat_type, i)))
    net.eval()

    # Get one sample per class that is correctly classified
    keep = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader, 0):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            keep.extend(predicted.eq(targets).cpu().numpy())
    keep = np.asarray(keep)
    idxs = []
    if args.dataset == 'cifar10':
        for c in range(max(testset.targets) + 1):
            idx = [j for j, label in enumerate(testset.targets) if label == c and keep[j] > 0]
            idxs.append(np.random.permutation(idx)[0])
    elif args.dataset == 'gtsrb':
        labels = None
        for batch_idx, (_, targets) in enumerate(testloader, 0):
            if labels is not None:
                labels = torch.cat([labels, targets])
            else:
                labels = targets
        labels = labels.numpy()
        for c in range(max(labels) + 1):
            idx = [j for j, label in enumerate(labels) if label == c and keep[j] > 0]
            idxs.append(np.random.permutation(idx)[0])
    else:
        for c in range(max(testset.labels) + 1):
            idx = [j for j, label in enumerate(testset.labels) if label == c and keep[j] > 0]
            idxs.append(np.random.permutation(idx)[0])

    # Compute LDP based on randomized smoothing
    probs = []
    for idx in idxs:

        image = testset.__getitem__(idx)[0]

        # Prob of image
        with torch.no_grad():
            outputs_unsmoothed = net(torch.unsqueeze(image.to(device), dim=0))
            _, predicted_unsmoothed = outputs_unsmoothed.max(1)

            image_smoothed = smooth(image.to(device), N=1024, sigma=args.sigma)
            outputs = net(image_smoothed)
            _, predicted = outputs.max(1)
            prob = get_freq(predicted.detach().cpu().numpy(), outputs.size(1))
            probs.append(prob)
    probs = np.asarray(probs)
    probs = np.mean(probs, axis=0)
    probs_max.append(np.amax(probs))
probs_max = np.asarray(probs_max)
np.save('stat_{}_{}_{}.npy'.format(args.dataset, args.stat_type, args.sigma), probs_max)
