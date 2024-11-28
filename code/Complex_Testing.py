import sys
import typer
# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
import os


# BatchEnsemble implementation

def random_sign_(tensor: torch.Tensor, prob: float = 0.5, value: float = 1.0):
    """
    Randomly set elements of the input tensor to either +value or -value.
    """
    sign = torch.where(torch.rand_like(tensor) < prob, 1.0, -1.0)
    with torch.no_grad():
        tensor.copy_(sign * value)


class BatchEnsembleMixin:
    def init_ensemble(
            self,
            in_features: int,
            out_features: int,
            ensemble_size: int,
            alpha_init: float | None = None,
            gamma_init: float | None = None,
            bias: bool = True,
            device=None,
            dtype=None,
    ):
        self.ensemble_size = ensemble_size
        self.alpha_init = alpha_init
        self.gamma_init = gamma_init

        if not isinstance(self, nn.Module):
            raise TypeError("BatchEnsembleMixin must be mixed with nn.Module or one of its subclasses")

        if alpha_init is None:
            self.register_parameter("alpha_param", None)
        else:
            self.alpha_param = self.init_scaling_parameter(alpha_init, in_features, device=device, dtype=dtype)
            self.register_parameter("alpha_param", self.alpha_param)

        if gamma_init is None:
            self.register_parameter("gamma_param", None)
        else:
            self.gamma_param = self.init_scaling_parameter(gamma_init, out_features, device=device, dtype=dtype)
            self.register_parameter("gamma_param", self.gamma_param)

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(ensemble_size, out_features, device=device, dtype=dtype))
            self.register_parameter("bias_param", self.bias_param)
        else:
            self.register_parameter("bias_param", None)

    def init_scaling_parameter(self, init_value: float, num_features: int, device=None, dtype=None):
        param = torch.empty(self.ensemble_size, num_features, device=device, dtype=dtype)
        if init_value < 0:
            param.normal_(mean=1, std=-init_value)
        else:
            random_sign_(param, prob=init_value, value=1.0)
        return nn.Parameter(param)

    def expand_param(self, x: torch.Tensor, param: torch.Tensor):
        num_repeats = x.size(0) // self.ensemble_size
        expanded_param = torch.repeat_interleave(param, num_repeats, dim=0)
        extra_dims = len(x.shape) - len(expanded_param.shape)
        for _ in range(extra_dims):
            expanded_param = expanded_param.unsqueeze(-1)
        return expanded_param


class BELinear(nn.Linear, BatchEnsembleMixin):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            ensemble_size: int = 1,
            alpha_init: float | None = None,
            gamma_init: float | None = None,
            device=None,
            dtype=None,
    ):
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)
        self.init_ensemble(in_features, out_features, ensemble_size, alpha_init, gamma_init, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.alpha_init is not None:
            input = input * self.expand_param(input, self.alpha_param)
        x = F.linear(input, self.weight)
        if self.gamma_init is not None:
            x = x * self.expand_param(x, self.gamma_param)
        if self.bias_param is not None:
            x = x + self.expand_param(x, self.bias_param)
        return x


class Conv2d(nn.Conv2d, BatchEnsembleMixin):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias: bool = True,
            padding_mode='zeros',
            ensemble_size: int = 1,
            alpha_init: float | None = None,
            gamma_init: float | None = None,
            device=None,
            dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,  # Bias is managed separately
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        self.init_ensemble(
            in_features=in_channels,
            out_features=out_channels,
            ensemble_size=ensemble_size,
            alpha_init=alpha_init,
            gamma_init=gamma_init,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.alpha_init is not None:
            input = input * self.expand_param(input, self.alpha_param)
        x = self._conv_forward(input, self.weight, None)
        if self.gamma_init is not None:
            x = x * self.expand_param(x, self.gamma_param)
        if self.bias_param is not None:
            x = x + self.expand_param(x, self.bias_param)
        return x


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(
            self,
            num_features: int,
            eps=1e-5,
            momentum=0.1,
            affine=False,  # We will manage affine parameters ourselves
            track_running_stats=True,
            device=None,
            dtype=None,
            ensemble_size: int = 1,
    ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype
        )
        self.ensemble_size = ensemble_size
        # Correct tensor initialization
        self.weight_be = nn.Parameter(torch.empty(self.ensemble_size, num_features, device=device, dtype=dtype))
        self.bias_be = nn.Parameter(torch.empty(self.ensemble_size, num_features, device=device, dtype=dtype))
        self.reset_be_parameters()

    def reset_be_parameters(self):
        nn.init.ones_(self.weight_be)
        nn.init.zeros_(self.bias_be)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Use the base class's forward method
        x = super().forward(input)
        num_repeats = x.size(0) // self.ensemble_size
        weight = torch.repeat_interleave(self.weight_be, num_repeats, dim=0).unsqueeze(2).unsqueeze(3)
        bias = torch.repeat_interleave(self.bias_be, num_repeats, dim=0).unsqueeze(2).unsqueeze(3)
        x = x * weight + bias
        return x


# StandardScaler
class StandardScaler:
    """
    Based on https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
    Extended with inverse_transform
    """

    def __init__(
            self,
            mean: torch.Tensor | None = None,
            std: torch.Tensor | None = None,
            epsilon: float = 1e-8,
    ):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, data: torch.Tensor):
        reduction_axes = list(range(data.dim() - 1))
        self.mean = torch.mean(data, dim=reduction_axes)
        self.std = torch.std(data, dim=reduction_axes)
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit before calling transform.")
        scaled_mean = (data - self.mean) / (self.std + self.epsilon)
        return scaled_mean

    def inverse_transform(self, scaled_data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit before calling inverse_transform.")
        unscaled_mean = scaled_data * (self.std + self.epsilon) + self.mean
        return unscaled_mean

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)

    def to(self, target_device: torch.device):
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.to(target_device)
            self.std = self.std.to(target_device)
        return self


# CNN Models
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.num_classes = 10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Changed input channels to 3
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 8 * 8, 10)  # Adjusted input size

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [batch_size, 32, 32, 32]
        x = self.pool(x)  # [batch_size, 32, 16, 16]
        x = self.relu(self.bn2(self.conv2(x)))  # [batch_size, 64, 16, 16]
        x = self.pool(x)  # [batch_size, 64, 8, 8]
        x = x.view(x.size(0), -1)  # [batch_size, 64*8*8]
        x = self.fc(x)  # [batch_size, 10]
        return x


class SharedParametersCNN(nn.Module):
    def __init__(self, num_heads=4):
        super(SharedParametersCNN, self).__init__()
        self.num_heads = num_heads
        self.num_classes = 10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Changed input channels to 3
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.shared_fc = nn.Linear(64 * 8 * 8, 128)  # Adjusted input size
        # Multiple heads for classification
        self.heads = nn.ModuleList([nn.Linear(128, 10) for _ in range(num_heads)])

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [batch_size, 32, 32, 32]
        x = self.pool(x)  # [batch_size, 32, 16, 16]
        x = self.relu(self.bn2(self.conv2(x)))  # [batch_size, 64, 16, 16]
        x = self.pool(x)  # [batch_size, 64, 8, 8]
        x = x.view(x.size(0), -1)  # [batch_size, 64*8*8]
        x = self.shared_fc(x)  # [batch_size, 128]
        # Collect outputs from all heads
        outputs = [head(x) for head in self.heads]
        outputs = torch.stack(outputs)  # Shape: [num_heads, batch_size, num_classes]
        # Average the outputs over heads
        x = outputs.mean(dim=0)  # Shape: [batch_size, num_classes]
        return x


class BatchEnsembleCNN(nn.Module):
    def __init__(self, ensemble_size=4, alpha_init=0.5, gamma_init=0.5):
        super(BatchEnsembleCNN, self).__init__()
        self.num_classes = 10
        self.ensemble_size = ensemble_size
        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1,  # Changed input channels to 3
                            ensemble_size=ensemble_size,
                            alpha_init=alpha_init,
                            gamma_init=gamma_init)
        self.bn1 = BatchNorm2d(32, ensemble_size=ensemble_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1,
                            ensemble_size=ensemble_size,
                            alpha_init=alpha_init,
                            gamma_init=gamma_init)
        self.bn2 = BatchNorm2d(64, ensemble_size=ensemble_size)
        self.fc = BELinear(64 * 8 * 8, 10,  # Adjusted input size
                           ensemble_size=ensemble_size,
                           alpha_init=alpha_init,
                           gamma_init=gamma_init)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [batch_size * ensemble_size, 32, 32, 32]
        x = self.pool(x)  # [batch_size * ensemble_size, 32, 16, 16]
        x = self.relu(self.bn2(self.conv2(x)))  # [batch_size * ensemble_size, 64, 16, 16]
        x = self.pool(x)  # [batch_size * ensemble_size, 64, 8, 8]
        x = x.view(x.size(0), -1)  # [batch_size * ensemble_size, 64*8*8]
        x = self.fc(x)  # [batch_size * ensemble_size, 10]
        return x


class SharedParametersBatchEnsembleCNN(nn.Module):
    def __init__(self, ensemble_size=4, alpha_init=0.5, gamma_init=0.5):
        super(SharedParametersBatchEnsembleCNN, self).__init__()
        self.ensemble_size = ensemble_size
        self.num_classes = 10
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        # Shared BatchEnsemble convolutional layers
        self.conv1 = Conv2d(
            3, 32, kernel_size=3, padding=1,  # Changed input channels to 3
            ensemble_size=ensemble_size,
            alpha_init=alpha_init,
            gamma_init=gamma_init
        )
        self.bn1 = BatchNorm2d(32, ensemble_size=ensemble_size)
        self.conv2 = Conv2d(
            32, 64, kernel_size=3, padding=1,
            ensemble_size=ensemble_size,
            alpha_init=alpha_init,
            gamma_init=gamma_init
        )
        self.bn2 = BatchNorm2d(64, ensemble_size=ensemble_size)

        # Shared fully connected layer
        self.shared_fc = BELinear(
            64 * 8 * 8, 128,  # Adjusted input size
            ensemble_size=ensemble_size,
            alpha_init=alpha_init,
            gamma_init=gamma_init
        )

        # Multiple heads for classification
        self.heads = nn.ModuleList([
            nn.Linear(128, self.num_classes) for _ in range(ensemble_size)
        ])

    def forward(self, x):
        # Shared convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))  # Shape: [batch_size * ensemble_size, 32, 32, 32]
        x = self.pool(x)  # Shape: [batch_size * ensemble_size, 32, 16, 16]
        x = self.relu(self.bn2(self.conv2(x)))  # Shape: [batch_size * ensemble_size, 64, 16, 16]
        x = self.pool(x)  # Shape: [batch_size * ensemble_size, 64, 8, 8]
        x = x.view(x.size(0), -1)  # Shape: [batch_size * ensemble_size, 64*8*8]

        # Shared fully connected layer
        x = self.shared_fc(x)  # Shape: [batch_size * ensemble_size, 128]

        # Reshape x to [ensemble_size, batch_size, 128]
        batch_size = x.size(0) // self.ensemble_size
        x = x.view(self.ensemble_size, batch_size, -1)

        # Apply each head and collect outputs
        outputs = []
        for i, head in enumerate(self.heads):
            out = head(x[i])  # x[i] is [batch_size, 128]
            outputs.append(out)

        # Stack outputs: Shape [ensemble_size, batch_size, num_classes]
        outputs = torch.stack(outputs)  # Shape: [ensemble_size, batch_size, num_classes]

        # Average over ensemble members
        outputs = outputs.mean(dim=0)  # Shape: [batch_size, num_classes]

        return outputs


class ComplexCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ComplexCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x3 -> 32x32x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32x64 -> 16x16x64

            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 16x16x64 -> 16x16x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x128 -> 8x8x128

            # Conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 8x8x128 -> 8x8x256
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8x256 -> 8x8x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x256 -> 4x4x256

            # Conv5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 4x4x256 -> 4x4x512
            nn.ReLU(inplace=True),

            # Conv6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 -> 4x4x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4x512 -> 2x2x512

            # Conv7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 2x2x512 -> 2x2x512
            nn.ReLU(inplace=True),

            # Conv8
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 2x2x512 -> 2x2x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2x512 -> 1x1x512
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class BatchEnsembleComplexCNN(nn.Module):
    def __init__(self, num_classes=10, ensemble_size=4, alpha_init=0.5, gamma_init=0.5):
        super(BatchEnsembleComplexCNN, self).__init__()
        self.num_classes = num_classes
        self.ensemble_size = ensemble_size

        # Define BatchEnsemble convolutional layers
        self.features = nn.Sequential(
            # Conv1
            Conv2d(3, 64, kernel_size=3, padding=1,  # Input: 32x32x3 -> Output: 32x32x64
                   ensemble_size=ensemble_size,
                   alpha_init=alpha_init,
                   gamma_init=gamma_init),
            BatchNorm2d(64, ensemble_size=ensemble_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16x16x64

            # Conv2
            Conv2d(64, 128, kernel_size=3, padding=1,  # Output: 16x16x128
                   ensemble_size=ensemble_size,
                   alpha_init=alpha_init,
                   gamma_init=gamma_init),
            BatchNorm2d(128, ensemble_size=ensemble_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 8x8x128

            # Conv3
            Conv2d(128, 256, kernel_size=3, padding=1,  # Output: 8x8x256
                   ensemble_size=ensemble_size,
                   alpha_init=alpha_init,
                   gamma_init=gamma_init),
            BatchNorm2d(256, ensemble_size=ensemble_size),
            nn.ReLU(inplace=True),

            # Conv4
            Conv2d(256, 256, kernel_size=3, padding=1,  # Output: 8x8x256
                   ensemble_size=ensemble_size,
                   alpha_init=alpha_init,
                   gamma_init=gamma_init),
            BatchNorm2d(256, ensemble_size=ensemble_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 4x4x256

            # Conv5
            Conv2d(256, 512, kernel_size=3, padding=1,  # Output: 4x4x512
                   ensemble_size=ensemble_size,
                   alpha_init=alpha_init,
                   gamma_init=gamma_init),
            BatchNorm2d(512, ensemble_size=ensemble_size),
            nn.ReLU(inplace=True),

            # Conv6
            Conv2d(512, 512, kernel_size=3, padding=1,  # Output: 4x4x512
                   ensemble_size=ensemble_size,
                   alpha_init=alpha_init,
                   gamma_init=gamma_init),
            BatchNorm2d(512, ensemble_size=ensemble_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 2x2x512

            # Conv7
            Conv2d(512, 512, kernel_size=3, padding=1,  # Output: 2x2x512
                   ensemble_size=ensemble_size,
                   alpha_init=alpha_init,
                   gamma_init=gamma_init),
            BatchNorm2d(512, ensemble_size=ensemble_size),
            nn.ReLU(inplace=True),

            # Conv8
            Conv2d(512, 512, kernel_size=3, padding=1,  # Output: 2x2x512
                   ensemble_size=ensemble_size,
                   alpha_init=alpha_init,
                   gamma_init=gamma_init),
            BatchNorm2d(512, ensemble_size=ensemble_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 1x1x512
        )

        # Define BatchEnsemble classifier
        self.classifier = nn.Sequential(
            BELinear(512 * 1 * 1, 512,
                     ensemble_size=ensemble_size,
                     alpha_init=alpha_init,
                     gamma_init=gamma_init),
            nn.ReLU(inplace=True),
            BELinear(512, num_classes,
                     ensemble_size=ensemble_size,
                     alpha_init=alpha_init,
                     gamma_init=gamma_init)
        )

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.features(x)  # Shape: [batch_size * ensemble_size, 1, 1, 512]

        # Flatten
        x = x.view(x.size(0), -1)  # Shape: [batch_size * ensemble_size, 512]

        # Forward pass through classifier
        x = self.classifier(x)  # Shape: [batch_size * ensemble_size, num_classes]

        return x


# Training and visualization functions

def train_mnist_model(
        model,
        train_loader,
        test_loader,
        model_type="simple",
        ensemble_size=4,
        num_epochs=15,
        lr=0.0002,
        device=torch.device("cpu"),
):
    model.to(device)  # Move the model to the GPU if available
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if model_type == "batchensemble" or model_type == "batchensemble_complex":
                # Repeat images and labels for BatchEnsemble
                images = images.repeat(ensemble_size, 1, 1, 1)
                labels = labels.repeat(ensemble_size)
            elif model_type == "shared_batchensemble":
                # Repeat images only
                images = images.repeat(ensemble_size, 1, 1, 1)
                # Do not repeat labels
            # For other model types (including "complex"), no repetition is needed
            outputs = model(images)
            if model_type == "shared_batchensemble":
                # Reshape outputs to [ensemble_size, batch_size, num_classes]
                outputs = outputs.view(ensemble_size, -1, model.num_classes)
                # Average the outputs over ensemble members
                outputs = outputs.mean(dim=0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                if model_type == "batchensemble" or model_type == "batchensemble_complex":
                    images = images.repeat(ensemble_size, 1, 1, 1)
                    outputs = model(images)
                    # Reshape and average over ensemble instances
                    outputs = outputs.view(ensemble_size, -1, model.num_classes)
                    outputs = outputs.mean(dim=0)
                elif model_type == "shared_batchensemble":
                    images = images.repeat(ensemble_size, 1, 1, 1)
                    outputs = model(images)
                else:
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], {model_type.replace("_", " ").capitalize()} Model Test Accuracy: {test_accuracy:.2f}%')

    return train_losses, test_accuracies


# Main function
def train_with_params(
        seed: int = typer.Option(42, help="Size of the seed"),
        ensemble_size: int = typer.Option(4, help="Size of the ensemble"),
        alpha_init: float = typer.Option(0.5, help="Initial value for Alpha in BatchEnsemble"),
        gamma_init: float = typer.Option(0.5, help="Initial value for Gamma in BatchEnsemble"),
        num_epochs: int = typer.Option(15, help="Number of training epochs"),
        batch_size: int = typer.Option(256, help="Batch size for training"),
        lr: float = typer.Option(0.0002, help="Learning rate for the optimizer"),
        device: str = typer.Option("cpu", help="Device to use, 'cuda' for GPU or 'cpu' for CPU")
):
    torch.manual_seed(seed)
    num_heads = ensemble_size  # For consistency

    # Ensure that batch_size is divisible by ensemble_size
    if batch_size % ensemble_size != 0:
        raise ValueError("Batch size must be divisible by ensemble size.")

    # Define transformations for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 mean
            std=[0.2023, 0.1994, 0.2010]  # CIFAR-10 std
        ),
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Define model types, including 'complex' and 'batchensemble_complex'
    model_types = [
        # "simple",
        # "batchensemble",
        # "sharedparameters",
        # "shared_batchensemble",
        # "complex",
        "batchensemble_complex"
    ]

    # Initialize dictionaries to store CNN models and their metrics
    cnn_models = {}
    cnn_train_losses = {}
    cnn_test_accuracies = {}

    # for the differences
    results = []

    # Train and evaluate CNN models
    for model_type in model_types:
        if model_type == "simple":
            model = SimpleCNN().to(device)
            ensemble_size_cnn = 1
        elif model_type == "batchensemble":
            model = BatchEnsembleCNN(
                ensemble_size=ensemble_size,
                alpha_init=alpha_init,
                gamma_init=gamma_init
            ).to(device)
            ensemble_size_cnn = ensemble_size
        elif model_type == "sharedparameters":
            model = SharedParametersCNN(
                num_heads=num_heads
            ).to(device)
            ensemble_size_cnn = 1  # Shared parameters handle multiple heads internally
        elif model_type == "shared_batchensemble":
            model = SharedParametersBatchEnsembleCNN(
                ensemble_size=ensemble_size,
                alpha_init=alpha_init,
                gamma_init=gamma_init
            ).to(device)
            ensemble_size_cnn = ensemble_size
        elif model_type == "complex":
            model = ComplexCNN().to(device)
            ensemble_size_cnn = 1  # No ensemble in this model
        elif model_type == "batchensemble_complex":
            model = BatchEnsembleComplexCNN(
                ensemble_size=ensemble_size,
                alpha_init=alpha_init,
                gamma_init=gamma_init
            ).to(device)
            ensemble_size_cnn = ensemble_size
        else:
            raise ValueError("Invalid model_type.")

        cnn_models[model_type] = model
        print(f"Training CNN model: {model_type}")
        tr_losses, te_accuracies = train_mnist_model(
            model,
            train_loader,
            test_loader,
            model_type=model_type,
            ensemble_size=ensemble_size_cnn,
            num_epochs=num_epochs,
            lr=lr,
            device=device,  # Ensure device is passed
        )

        cnn_train_losses[model_type] = tr_losses
        cnn_test_accuracies[model_type] = te_accuracies

        result = {
            "seed": seed,
            "alpha_init": alpha_init,
            "gamma_init": gamma_init,
            "ensemble_size": ensemble_size,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "model_type": model_type,
            "final_test_accuracy": te_accuracies[-1],
            "avg_test_accuracy": sum(te_accuracies) / len(te_accuracies)
        }
        results.append(result)

    return results


def save_results_to_csv(results, file_path):
    """Saves results to a CSV file without overwriting existing data."""
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Check if file exists
    file_exists = os.path.isfile(file_path)

    # Append data to the file
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)

    if file_exists:
        print(f"Appended results to {file_path}")
    else:
        print(f"Created new results file: {file_path}")


app = typer.Typer()


@app.command()
def run_experiments():
    """Defines a CLI to test multiple parameter combinations."""
    alpha_list = [-0.1, 0.5, 1]
    gamma_list = [-0.1, 0.5, 1]
    ensemble_size_list = [2, 5, 10]
    seed_list = [42]
    epochs_list = [15]
    batch_list = [256]
    lr_list = [0.0002]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)

    alpha_gamma_file = os.path.join(output_folder, "alpha_gamma_results.csv")
    ensemble_size_file = os.path.join(output_folder, "ensemble_size_results.csv")

    """
    # test different alpha and gamma values
    for alpha in alpha_list:
        for gamma in gamma_list:
            print(f"Next calculation with differences in alpha and gamma: alpha={alpha} and gamma={gamma}")
            results = train_with_params(seed=42, ensemble_size=4, alpha_init=alpha, gamma_init=gamma, num_epochs=15, batch_size=256, lr=0.0002, device=device)
            save_results_to_csv(results, alpha_gamma_file)
    """

    # test ensemble size
    for ensemble_size in ensemble_size_list:
        print(f"Next calculation with differences in ensemble size: ensemble size = {ensemble_size}")
        results = train_with_params(seed=42, ensemble_size=ensemble_size, alpha_init=0.5, gamma_init=0.5, num_epochs=15,
                                    batch_size=ensemble_size, lr=0.0002, device=device)
        save_results_to_csv(results, ensemble_size_file)


sys.argv = ["Complex_Testing.py"]
typer.run(run_experiments)