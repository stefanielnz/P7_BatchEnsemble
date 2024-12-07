from typing import Iterable

from typing_extensions import Self

import torch


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
        """Standard Scaler.

        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit_batched(self, data_iterable: Iterable[torch.Tensor]) -> Self:
        sum_ = None
        sum_of_squares = None
        count = 0

        for data in data_iterable:  # [batch_size, num_features]
            reduction_axes = list(range(data.dim() - 1))
            batch_count = data.size(0)

            # Update total count of samples
            count += batch_count

            # Calculate batch sum and sum of squares
            batch_sum = torch.sum(data, dim=reduction_axes)  # [num_features]
            batch_sum_of_squares = torch.sum(data**2, dim=reduction_axes)  # [num_features]

            # Update running sum and sum of squares
            if sum_ is None:
                sum_ = batch_sum
                sum_of_squares = batch_sum_of_squares
            else:
                sum_ += batch_sum
                sum_of_squares += batch_sum_of_squares

        if sum_ is None or sum_of_squares is None:
            raise RuntimeError("No data provided.")

        # Compute mean and std
        mean = sum_ / count  # [num_features]
        var = (sum_of_squares / count) - (mean**2)  # [num_features]
        if count > 1:
            var *= count / (count - 1)
        else:  # provide nan if only one sample is provided (similar to torch.sqrt)
            var = torch.full_like(var, float("nan"))
        std = torch.sqrt(var)

        self.mean = mean
        self.std = std

        return self

    def fit(self, data: torch.Tensor) -> Self:
        reduction_axes = list(range(data.dim() - 1))
        print(reduction_axes)
        self.mean = torch.mean(data, dim=reduction_axes)
        self.std = torch.std(data, dim=reduction_axes)
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit before calling transform.")
        scaled_mean = (data - self.mean) / (self.std + self.epsilon)
        return scaled_mean

    def transform_std(self, std: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit before calling transform.")
        scaled_std = std / (self.std + self.epsilon)
        return scaled_std

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, scaled_data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit before calling inverse_transform.")
        unscaled_mean = scaled_data * (self.std + self.epsilon) + self.mean
        return unscaled_mean

    def inverse_transform_std(self, scaled_std: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit before calling inverse_transform.")
        unscaled_std = scaled_std * (self.std + self.epsilon)
        return unscaled_std

    def to(self, target_device: torch.device) -> Self:
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.to(target_device)
            self.std = self.std.to(target_device)
        return self
