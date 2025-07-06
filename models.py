import torch
import torch.nn as nn

CUDA = torch.cuda.is_available()


def CORAL(source, target):
    """
    Compute the CORAL loss between source and target feature representations.
    
    Args:
        source (Tensor): Feature tensor from source domain. Shape: (batch_size, features)
        target (Tensor): Feature tensor from target domain. Shape: (batch_size, features)
        
    Returns:
        loss (Tensor): CORAL loss value.
    """
    d = source.data.shape[1]

    # Source covariance
    xm = source - torch.mean(source, dim=0, keepdim=True)
    xc = xm.t() @ xm

    # Target covariance
    xmt = target - torch.mean(target, dim=0, keepdim=True)
    xct = xmt.t() @ xmt

    # Frobenius norm between source and target covariances
    loss = torch.mean((xc - xct) ** 2)
    loss = loss / (4 * d * d)

    return loss


class DeepCORAL(nn.Module):
    """
    Deep CORAL model combining a shared AlexNet and a task-specific classifier.
    """
    def __init__(self, num_classes=1000):
        super(DeepCORAL, self).__init__()
        self.sharedNet = AlexNet()
        self.fc = nn.Linear(4096, num_classes)

        # Initialize classifier weights as in the original CORAL paper
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        """
        Forward pass for both source and target data.
        
        Returns:
            source_out, target_out: Classifier outputs for both domains.
        """
        source = self.sharedNet(source)
        source = self.fc(source)

        target = self.sharedNet(target)
        target = self.fc(target)

        return source, target


class AlexNet(nn.Module):
    """
    Modified AlexNet feature extractor used in Deep CORAL.
    """
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of AlexNet for feature extraction.
        """
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
