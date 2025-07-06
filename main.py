from __future__ import division
import argparse
import torch
from torch.utils import model_zoo
from torch.autograd import Variable

import models
import utils
from data_loader import get_train_test_loader, get_fabric_dataloader

# Configuration
CUDA = torch.cuda.is_available()
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
BATCH_SIZE = [32, 32]
EPOCHS = 10

# Load datasets
source_loader = get_fabric_dataloader(case='color', batch_size=BATCH_SIZE[0])
target_loader = get_fabric_dataloader(case='grayscale', batch_size=BATCH_SIZE[1])


def train(model, optimizer, epoch, _lambda):
    """
    Train the model for one epoch.
    """
    result = []

    source = list(enumerate(source_loader))
    target = list(enumerate(target_loader))
    train_steps = min(len(source), len(target))

    for batch_idx in range(train_steps):
        _, (source_data, source_label) = source[batch_idx]
        _, (target_data, _) = target[batch_idx]

        if CUDA:
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        optimizer.zero_grad()
        out1, out2 = model(source_data, target_data)

        classification_loss = torch.nn.functional.cross_entropy(out1, source_label)
        coral_loss = models.CORAL(out1, out2)

        sum_loss = _lambda * coral_loss + classification_loss
        sum_loss.backward()
        optimizer.step()

        result.append({
            'epoch': epoch,
            'step': batch_idx + 1,
            'total_steps': train_steps,
            'lambda': _lambda,
            'coral_loss': coral_loss.item(),
            'classification_loss': classification_loss.item(),
            'total_loss': sum_loss.item()
        })

        print(
            f'Train Epoch: {epoch:2d} [{batch_idx + 1:2d}/{train_steps}]\t'
            f'Lambda: {_lambda:.4f}, Class: {classification_loss.item():.6f}, '
            f'CORAL: {coral_loss.item():.6f}, Total_Loss: {sum_loss.item():.6f}'
        )

    return result


def test(model, dataset_loader, epoch):
    """
    Evaluate the model on a dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in dataset_loader:
            if CUDA:
                data, target = data.cuda(), target.cuda()

            out, _ = model(data, data)

            test_loss += torch.nn.functional.cross_entropy(out, target, reduction='sum').item()
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(dataset_loader.dataset)

    return {
        'epoch': epoch,
        'average_loss': test_loss,
        'correct': correct,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * correct / len(dataset_loader.dataset)
    }


def load_pretrained(model):
    """
    Load ImageNet pre-trained weights into the sharedNet part of the model.
    """
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='Resume from checkpoint file')
    args = parser.parse_args()

    model = models.DeepCORAL(num_classes=4)  # Adjust if different number of classes

    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.fc.parameters(), 'lr': 10 * LEARNING_RATE},
    ], lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    if CUDA:
        model = model.cuda()

    if args.load:
        utils.load_net(model, args.load)
    else:
        load_pretrained(model.sharedNet)

    training_statistic = []
    testing_s_statistic = []
    testing_t_statistic = []
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        _lambda = (epoch + 1) / EPOCHS

        res = train(model, optimizer, epoch + 1, _lambda)
        training_statistic.append(res)

        print(f'### EPOCH {epoch + 1}: '
              f'Class: {sum(r["classification_loss"] / r["total_steps"] for r in res):.6f}, '
              f'CORAL: {sum(r["coral_loss"] / r["total_steps"] for r in res):.6f}, '
              f'Total: {sum(r["total_loss"] / r["total_steps"] for r in res):.6f}')

        test_source = test(model, source_loader, epoch)
        test_target = test(model, target_loader, epoch)

        testing_s_statistic.append(test_source)
        testing_t_statistic.append(test_target)

        print(f'### Test Source: Epoch {epoch + 1}, '
              f'Loss: {test_source["average_loss"]:.4f}, '
              f'Accuracy: {test_source["correct"]}/{test_source["total"]} '
              f'({test_source["accuracy"]:.2f}%)')

        print(f'### Test Target: Epoch {epoch + 1}, '
              f'Loss: {test_target["average_loss"]:.4f}, '
              f'Accuracy: {test_target["correct"]}/{test_target["total"]} '
              f'({test_target["accuracy"]:.2f}%)')

        if test_target['accuracy'] > best_accuracy:
            best_accuracy = test_target['accuracy']
            print(f'New best target accuracy: {best_accuracy:.2f}%. Saving model...')
            utils.save_net(model, 'best_model_checkpoint.tar')

    # Save training/testing statistics
    utils.save(training_statistic, 'training_statistic.pkl')
    utils.save(testing_s_statistic, 'testing_s_statistic.pkl')
    utils.save(testing_t_statistic, 'testing_t_statistic.pkl')
