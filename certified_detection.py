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
from scipy import special
from scipy.stats import norm
from scipy.stats import entropy

from src.util import data_split, smooth, get_freq
from src.model_zoo import ShadowModel, SimpleNet, MobileNet
from src.gtsrb import GTSRB


parser = argparse.ArgumentParser(description='Certified detection')
parser.add_argument('--dataset', type=str, required=True, help='cifar10, svhn, gtsrb')
parser.add_argument('--sigma', type=float, default=2.0, help='Standard deviation of the Gaussian noise')
parser.add_argument('--beta', type=float, default=0.2, help='Proportion of outliers removed from the calibration set')
parser.add_argument('--theta', type=float, default=0.05, help='Significance level')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(datetime.now())

# Obtain the detection threshold based on the calibration set
stats_null = np.load('stat_{}_shadow_{}.npy'.format(args.dataset, args.sigma))
stats_null_ranked = np.sort(stats_null)
N = len(stats_null)
m = int(N * args.beta)
idx_thres = int(N - m - np.floor((N - m + 1) * args.theta))
thres = stats_null_ranked[idx_thres]

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

# TPR and CTPR
actual = 0
certified = 0
total = 0
for i in range(len(os.listdir('{}_attack'.format(args.dataset)))):

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
    net.load_state_dict(torch.load('{}_attack/model_{}.pth'.format(args.dataset, i)))
    net.eval()

    # Load trigger
    trigger = torch.load('{}_trigger/trigger_{}'.format(args.dataset, i))
    target_class = torch.load('{}_trigger/target_class_{}'.format(args.dataset, i))

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

    strs = []
    deltas = []
    prob_actual_aggre = 0
    for idx in idxs:
        image = testset.__getitem__(idx)[0]
        image_bd = torch.clamp(image + trigger, min=0, max=1)

        # Prob of image
        with torch.no_grad():
            # Compute SLPV for the image
            image_smoothed = smooth(image.to(device), N=1024, sigma=args.sigma)
            outputs = net(image_smoothed)
            _, predicted = outputs.max(1)
            prob = get_freq(predicted.detach().cpu().numpy(), outputs.size(1))
            prob_actual_aggre += prob[target_class]

            # Compute STR and delta
            image_bd_smoothed = smooth(image_bd.to(device), N=1024, sigma=args.sigma)
            outputs_bd = net(image_bd_smoothed)
            _, predicted_bd = outputs_bd.max(1)
            prob_bd = get_freq(predicted_bd.detach().cpu().numpy(), outputs_bd.size(1))
            deltas.append(torch.norm(image_bd - image).detach().cpu().numpy())
            strs.append(prob_bd[target_class])
            
    strs = np.asarray(strs)
    deltas = np.asarray(deltas)
    Pi = np.min(strs)
    Delta = np.max(deltas)
    if sigma * (Ginv(1 - thres) - Ginv(1 - pi)) - Delta > 0:
        certified += 1
    prob_actual_aggre /= len(idxs)
    if prob_actual_aggre > thres:
        actual += 1
    total += 1
print('CTPR: %.3f' % (certified / total))
print('TPR: %.3f' % (actual / total))

# FPR
false_detection = 0
total = 0
for i in range(len(os.listdir('{}_clean'.format(args.dataset)))):

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
    net.load_state_dict(torch.load('./{}_clean/model_{}.pth'.format(args.dataset, i)))
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
    if np.amax(probs) > thres:
        false_detection += 1
    total += 1
print('FPR: %.3f' % (false_detection / total))
