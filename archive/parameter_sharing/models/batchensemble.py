import torch
import torch.nn as nn
import torch.nn.functional as F


def random_sign_(tensor: torch.Tensor, prob: float = 0.5, value: float = 1.0):
    """
    Randomly set elements of the input tensor to either +value or -value.

    Args:
        tensor (torch.Tensor): Input tensor.
        prob (float, optional): Probability of setting an element to +value (default: 0.5).
        value (float, optional): Value to set the elements to (default: 1.0).

    Returns:
        torch.Tensor: Tensor with elements set to +value or -value.
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
        """Expand and match a parameter to a given input tensor.

        Description:
        In BatchEnsemble, the alpha, gamma and bias parameters are expanded to match the input tensor.

        Args:
            x: Input tensor to match the parameter to. Shape: [batch_size, features/classes, ...]
            param: Parameter to expand. Shape: [ensemble_size, features/classes]

        Returns:
            expanded_param: Expanded parameter. Shape: [batch_size, features/classes, ...]
        """
        num_repeats = x.size(0) // self.ensemble_size
        expanded_param = torch.repeat_interleave(param, num_repeats, dim=0)
        extra_dims = len(x.shape) - len(expanded_param.shape)
        for _ in range(extra_dims):
            expanded_param = expanded_param.unsqueeze(-1)
        return expanded_param


class Linear(nn.Linear, BatchEnsembleMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        ensemble_size: int = 1,
        alpha_init: float | None = None,
        gamma_init: float | None = None,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=False, device=device, dtype=dtype)
        # nn.init.kaiming_normal_(self.weight)
        self.init_ensemble(in_features, out_features, ensemble_size, alpha_init, gamma_init, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        self.alpha, self.gamma, and self.bias are applied to the input tensor x.
        If their params are None, the input tensor is returned unchanged.
        """
        if self.alpha_init is not None:
            input = input * self.expand_param(input, self.alpha_param)
        x = F.linear(input, self.weight)
        if self.gamma_init is not None:
            x = x * self.expand_param(x, self.gamma_param)
        if self.bias_param is not None:
            x = x + self.expand_param(x, self.bias_param)
        return x


class Conv2d(nn.Conv2d, BatchEnsembleMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Conv2d class is not implemented yet")


class BatchNorm2d(nn.BatchNorm2d, BatchEnsembleMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("BatchNorm2d class is not implemented yet")

    # Not same batchnorm
