import sys
import torch
import numpy as np
from torch.utils.data import Dataset


def data_split(dataset_name, dataset, type, ratio):

    if dataset_name == 'gtsrb':
        labels = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
        for batch_idx, (_, targets) in enumerate(dataloader, 0):
            if labels is not None:
                labels = torch.cat([labels, targets])
            else:
                labels = targets
        labels = labels.numpy()
    elif dataset_name == 'cifar10':
        labels = dataset.targets
    elif dataset_name == 'svhn':
        labels = dataset.labels
    else:
        sys.exit('Incorrect dataset name!')

    ind_keep = []
    num_classes = int(max(labels) + 1)
    for c in range(num_classes):
        ind = [i for i, label in enumerate(labels) if label == c]
        split = int(len(ind) * ratio)
        if type == 'evaluation':
            ind = ind[:split]
        elif type == 'defense':
            ind = ind[split:]
        else:
            sys.exit("Wrong training type!")
        ind_keep = ind_keep + ind

    if dataset_name == 'gtsrb':
        dataset._samples = [dataset._samples[i] for i in ind_keep]
    elif dataset_name == 'cifar10':
        dataset.data = dataset.data[ind_keep]
        dataset.targets = [dataset.targets[i] for i in ind_keep]
    elif dataset_name == 'svhn':
        dataset.data = dataset.data[ind_keep]
        dataset.labels = [dataset.labels[i] for i in ind_keep]

    return dataset


def smooth(images, N, sigma, device='cuda'):

    if len(images.size()) == 3:
        images = torch.unsqueeze(images, dim=0)

    if N == 0:
        return images

    images = images.repeat(N, 1, 1, 1)
    noise = torch.normal(0, sigma, size=images.size())

    return images + noise.to(device)


def get_freq(x, k):

    count = []
    for c in range(k):
        count.append(len(np.where(x == c)[0]))

    return np.asarray(count) / len(x)


def backdoor_embedding(image, trigger):

    image += trigger
    image *= 255
    image = image.round()
    image /= 255
    image = image.clamp(0, 1)

    return image


class AttackDataset(Dataset):

    def __init__(self, image, label):
        self.image = image
        self.label = label
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        return self.image[idx], self.label[idx]
