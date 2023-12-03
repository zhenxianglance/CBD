from __future__ import absolute_import
from __future__ import print_function

import torch
import sys
import numpy as np
import copy as cp
import random
from datetime import datetime
import sys

from src.util import backdoor_embedding, AttackDataset

random.seed(datetime.now())


def create_attack(dataset_name, trainset, testset):

    # The numbers are selected such that created trigger pattern will lead to a successful attack while having a constrained perturbation norm
    if dataset_name == 'cifar10':
        para = np.random.uniform(low=2, high=5, size=())
    elif dataset_name == 'gtsrb':
        para = np.random.randint(low=7, high=13, size=())
    elif dataset_name == 'svhn':
        para = np.random.randint(low=5, high=7, size=())
    else:
        sys.exit('Invalid dataset!')

    im_size = trainset.__getitem__(0)[0].size()

    if dataset_name == 'cifar10':
        pert_size = para / 255
        trigger = torch.zeros(im_size)
        for i in range(im_size[1]):
            for j in range(im_size[2]):
                if (i + j) % 2 == 0:
                    trigger[:, i, j] = torch.ones(im_size[0])
        trigger *= pert_size
    elif dataset_name == 'gtsrb':
        trigger = torch.randint(low=0, high=para + 1, size=(im_size[1], im_size[2])) / 255
        trigger = torch.unsqueeze(trigger, dim=0).repeat(im_size[0], 1, 1)
        mask = torch.ones(size=(im_size[1], im_size[2])) * 0.25
        mask = torch.bernoulli(mask)
        mask = torch.unsqueeze(mask, dim=0).repeat(im_size[0], 1, 1)
        trigger = trigger * mask
    elif dataset_name == 'svhn':
        trigger = (torch.randint(low=0, high=2, size=im_size) * 2 - 1) * para / 255
        mask = torch.ones(size=(im_size[1], im_size[2])) * 0.5
        mask = torch.bernoulli(mask)
        for i in range(im_size[1]):
            for j in range(im_size[2]):
                if i > int(im_size[1] / 2) or j > int(im_size[2] / 2):
                    mask[i, j] = 0
        mask = torch.unsqueeze(mask, dim=0).repeat(im_size[0], 1, 1)
        trigger = trigger * mask
    else:
        sys.exit('Invalid dataset!')
    # Normalize trigger
    if torch.norm(trigger) > 0.75:
        trigger /= 0.75

    # Choose target class
    if dataset_name == 'cifar10':
        n_class = max(trainset.targets) + 1
    elif dataset_name == 'gtsrb':
        labels_train = None
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)
        for batch_idx, (_, targets) in enumerate(trainloader, 0):
            if labels_train is not None:
                labels_train = torch.cat([labels_train, targets])
            else:
                labels_train = targets
        labels_train = labels_train.numpy()
        labels_test = None
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        for batch_idx, (_, targets) in enumerate(testloader, 0):
            if labels_test is not None:
                labels_test = torch.cat([labels_test, targets])
            else:
                labels_test = targets
        labels_test = labels_test.numpy()
        n_class = np.amax(labels_train)
    elif dataset_name == 'svhn':
        n_class = max(trainset.labels) + 1
    else:
        sys.exit('Invalid dataset!')
    target_class = np.random.choice(np.arange(start=0, stop=n_class, step=1, dtype=int))

    # Poisoning
    if dataset_name == 'cifar10':
        n_poison = 250
    elif dataset_name == 'gtsrb':
        n_poison = 50
    elif dataset_name == 'svhn':
        n_poison = 500
    else:
        sys.exit('Invalid dataset!')

    trainset_poisoned = cp.deepcopy(trainset)
    if dataset_name == 'gtsrb':
        train_images_attacks = None
        train_labels_attacks = None
    ind_train = []
    for c in range(n_class):
        if c == target_class:
            continue
        if dataset_name == 'cifar10':
            ind = [i for i, label in enumerate(trainset_poisoned.targets) if label == c]
        elif dataset_name == 'gtsrb':
            ind = [i for i, label in enumerate(labels_train) if label == c]
        elif dataset_name == 'svhn':
            ind = [i for i, label in enumerate(trainset_poisoned.labels) if label == c]
        else:
            sys.exit('Invalid dataset!')
        ind = np.random.choice(ind, np.amax([n_poison, len(ind)]), False)
        ind_train.extend(ind)
    if dataset_name == 'cifar10':
        for i in ind_train:
            image_poisoned = backdoor_embedding(image=trainset_poisoned.__getitem__(i)[0], trigger=trigger)
            trainset_poisoned.data = np.concatenate([trainset_poisoned.data, np.expand_dims(
                np.transpose((image_poisoned * 255).numpy(), [1, 2, 0]).astype(np.uint8), axis=0)], axis=0)
            trainset_poisoned.targets = np.concatenate(
                [trainset_poisoned.targets, np.expand_dims(target_class, axis=0)], axis=0)
    elif dataset_name == 'gtsrb':
        for i in ind_train:
            image_poisoned = backdoor_embedding(image=trainset.__getitem__(i)[0], trigger=trigger).unsqueeze(0)
            label_poisoned = torch.tensor([target_class], dtype=torch.long)
            if train_images_attacks is not None:
                train_images_attacks = torch.cat([train_images_attacks, image_poisoned], dim=0)
                train_labels_attacks = torch.cat([train_labels_attacks, label_poisoned], dim=0)
            else:
                train_images_attacks = image_poisoned
                train_labels_attacks = label_poisoned
        train_attack_data = AttackDataset(train_images_attacks, list(train_labels_attacks.numpy()))
        trainset_poisoned = torch.utils.data.ConcatDataset([trainset, train_attack_data])
    elif dataset_name == 'svhn':
        for i in ind_train:
            image_poisoned = backdoor_embedding(image=trainset_poisoned.__getitem__(i)[0], trigger=trigger)
            trainset_poisoned.data = np.concatenate(
                [trainset_poisoned.data, np.expand_dims((image_poisoned * 255).numpy().astype(np.uint8), axis=0)],
                axis=0)
            trainset_poisoned.labels = np.concatenate([trainset_poisoned.labels, np.expand_dims(target_class, axis=0)],
                                                      axis=0)
    else:
        sys.exit('Invalid dataset!')

    test_images_attacks = None
    test_labels_attacks = None
    if dataset_name == 'cifar10':
        ind_test = [i for i, label in enumerate(testset.targets) if label not in [target_class]]
        for i in ind_test:
            if test_images_attacks is not None:
                test_images_attacks = torch.cat(
                    [test_images_attacks, backdoor_embedding(image=testset.__getitem__(i)[0],
                                                             trigger=trigger).unsqueeze(0)], dim=0)
                test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([target_class], dtype=torch.long)],
                                                dim=0)
            else:
                test_images_attacks = backdoor_embedding(image=testset.__getitem__(i)[0], trigger=trigger).unsqueeze(0)
                test_labels_attacks = torch.tensor([target_class], dtype=torch.long)
    elif dataset_name == 'gtsrb':
        ind_test = [i for i, label in enumerate(labels_test) if label not in [target_class]]
        for i in ind_test:
            image_poisoned = backdoor_embedding(image=testset.__getitem__(i)[0], trigger=trigger).unsqueeze(0)
            label_poisoned = torch.tensor([target_class], dtype=torch.long)
            if test_images_attacks is not None:
                test_images_attacks = torch.cat([test_images_attacks, image_poisoned], dim=0)
                test_labels_attacks = torch.cat([test_labels_attacks, label_poisoned], dim=0)
            else:
                test_images_attacks = image_poisoned
                test_labels_attacks = label_poisoned
    elif dataset_name == 'svhn':
        ind_test = [i for i, label in enumerate(testset.labels) if label not in [target_class]]
        for i in ind_test:
            if test_images_attacks is not None:
                test_images_attacks = torch.cat(
                    [test_images_attacks, backdoor_embedding(image=testset.__getitem__(i)[0],
                                                             trigger=trigger, ).unsqueeze(0)], dim=0)
                test_labels_attacks = torch.cat([test_labels_attacks, torch.tensor([target_class], dtype=torch.long)],
                                                dim=0)
            else:
                test_images_attacks = backdoor_embedding(image=testset.__getitem__(i)[0], trigger=trigger).unsqueeze(0)
                test_labels_attacks = torch.tensor([target_class], dtype=torch.long)
    else:
        sys.exit('Invalid dataset!')
    attackset = torch.utils.data.TensorDataset(test_images_attacks, test_labels_attacks)

    return trainset_poisoned, attackset, trigger, target_class

