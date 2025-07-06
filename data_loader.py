import numpy as np
from torch.utils import data
from torchvision import datasets, transforms

import utils


def get_train_test_loader(directory, batch_size, testing_size=0.1, img_size=None):
    """
    Split a dataset into training and testing datasets.

    Args:
        directory (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        testing_size (float): Proportion of the dataset to use as the test set.
        img_size (tuple, optional): Resize images to this size (H, W).

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    mean, std = utils.get_dataset_mean_and_std(directory)

    transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if img_size is not None:
        transform.insert(0, transforms.Resize(img_size))

    dataset = datasets.ImageFolder(
        directory,
        transform=transforms.Compose(transform)
    )

    num_data = len(dataset)
    indices = list(range(num_data))
    split = int(np.floor(testing_size * num_data))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data.sampler.SubsetRandomSampler(train_idx),
        num_workers=2,
        drop_last=True
    )

    test_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data.sampler.SubsetRandomSampler(test_idx),
        num_workers=2,
        drop_last=True
    )

    return train_loader, test_loader


def get_fabric_dataloader(case, batch_size):
    """
    DataLoader for the Fabric dataset (color/grayscale).

    Args:
        case (str): One of ['color', 'grayscale'].
        batch_size (int): Number of samples per batch.

    Returns:
        data_loader (DataLoader): The data loader.
    """
    print(f'[INFO] Loading dataset: {case}')

    datas = {
        'color': 'dataset/fabric/color',
        'grayscale': 'dataset/fabric/grayscale'
    }

    means = {
        'color': [0.4411, 0.4729, 0.5579],
        'grayscale': [0.4790, 0.4790, 0.4790],
        'imagenet': [0.485, 0.456, 0.406]
    }

    stds = {
        'color': [0.1818, 0.1699, 0.1836],
        'grayscale': [0.1645, 0.1645, 0.1645],
        'imagenet': [0.229, 0.224, 0.225]
    }

    if case not in datas:
        raise ValueError(f"Unknown case '{case}'. Expected 'color' or 'grayscale'.")

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(means[case], stds[case]),
    ])

    data_loader = data.DataLoader(
        dataset=datasets.ImageFolder(datas[case], transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    return data_loader
