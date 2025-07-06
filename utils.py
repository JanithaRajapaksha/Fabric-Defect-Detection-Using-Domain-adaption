import pickle
import torch
from torch.utils import data
from torchvision import datasets, transforms


def get_dataset_mean_and_std(directory):
    """
    Compute the per-channel mean and standard deviation of images in a dataset folder.

    Args:
        directory (str): Path to the dataset folder.

    Returns:
        (list, list): Mean and standard deviation for each channel (RGB).
    """
    dataset = datasets.ImageFolder(
        directory,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    data_loader = data.DataLoader(dataset)

    mean = [0.0, 0.0, 0.0]
    std = [0.0, 0.0, 0.0]

    for channel in range(3):
        channel_sum = 0.0
        channel_std = 0.0

        for xs, _ in data_loader:
            img = xs[0][channel].numpy()
            channel_sum += img.mean()
            channel_std += img.std()

        mean[channel] = channel_sum / len(dataset)
        std[channel] = channel_std / len(dataset)

    return mean, std


def save(obj, path):
    """
    Save a Python object to a file using pickle.

    Args:
        obj: Python object to serialize.
        path (str): Destination file path.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'[INFO] Object saved to {path}')


def save_net(model, path):
    """
    Save a PyTorch model checkpoint.

    Args:
        model (torch.nn.Module): Model to save.
        path (str): Path to save the checkpoint.
    """
    torch.save(model.state_dict(), path)
    print(f'[INFO] Checkpoint saved to {path}')


def load_net(model, path):
    """
    Load a PyTorch model checkpoint.

    Args:
        model (torch.nn.Module): Model instance to load state dict into.
        path (str): Path to the saved checkpoint file.
    """
    model.load_state_dict(torch.load(path))
    print(f'[INFO] Checkpoint {path} loaded')
