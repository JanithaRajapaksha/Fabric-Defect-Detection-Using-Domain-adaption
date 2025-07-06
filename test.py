from __future__ import division
import argparse
import torch
from torch.utils import model_zoo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import models
import utils
from data_loader import get_fabric_dataloader

# CUDA availability
CUDA = torch.cuda.is_available()

# Load the same loaders used during training
source_loader = get_fabric_dataloader(case='color', batch_size=32)
target_loader = get_fabric_dataloader(case='grayscale', batch_size=32)


def test(model, dataset_loader, epoch):
    """
    Evaluate the model and print confusion matrix + classification report.
    """
    model.eval()
    test_loss = 0
    correct = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data, target in dataset_loader:
            if CUDA:
                data, target = data.cuda(), target.cuda()

            out, _ = model(data, data)
            test_loss += torch.nn.functional.cross_entropy(out, target, reduction='sum').item()

            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            true_labels.extend(target.cpu().numpy())
            predicted_labels.extend(pred.cpu().numpy())

    test_loss /= len(dataset_loader.dataset)

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_names = list(dataset_loader.dataset.class_to_idx.keys())
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    plot_confusion_matrix(conf_matrix, classes=class_names)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

    return {
        'epoch': epoch,
        'average_loss': test_loss,
        'correct': correct,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * correct / len(dataset_loader.dataset)
    }


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Display a confusion matrix as a heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='best_model_checkpoint.tar',
                        help='Path to the model checkpoint')
    args = parser.parse_args()

    # Load model
    model = models.DeepCORAL(num_classes=4)
    if CUDA:
        model = model.cuda()

    utils.load_net(model, args.checkpoint)

    # Run evaluation on target domain
    print("[INFO] Evaluating model on target domain (grayscale fabric)...")
    results = test(model, target_loader, epoch=0)
    print(f"\n[INFO] Final Test Accuracy: {results['accuracy']:.2f}%")
