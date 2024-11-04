import torch


def target_function(x: torch.Tensor):
    return x * torch.sin(x)


def scale_function(x: torch.Tensor, heteroscedastic: bool = True):
    """Calculates the scale of the noise."""
    scale = 0.1 + torch.abs(0.5 * x)
    return scale if heteroscedastic else torch.ones_like(x) * scale.mean()


def get_skafte_data(
    num_samples: int = 100,
    x_bounds: tuple[float, float] = (2.5, 12.5),
    heteroscedastic: bool = True,
):
    """Generates synthetic data with homoscedastic or heteroscedastic noise.

    Args:
      num_samples: The number of data points to generate.
      x_bounds: The lower and upper bounds of the x-values.
      heteroscedastic: Whether to use heteroscedastic noise.

    Returns:
      A tuple of tensors (x, y) representing the data.
    """
    x = torch.rand(num_samples) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    y = target_function(x) + torch.randn_like(x) * scale_function(x, heteroscedastic)
    return x.unsqueeze(1), y.unsqueeze(1)


if __name__ == "__main__":
    """Generates and plots synthetic data with both heteroscedastic and homoscedastic noise."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for i, heteroscedastic in enumerate([True, False]):
        x, y = get_skafte_data(heteroscedastic=heteroscedastic)

        # Sort x and y for plotting
        sorted_indices = torch.argsort(x[:, 0])
        x = x[sorted_indices].squeeze()
        y = y[sorted_indices].squeeze()

        y_std = scale_function(x, heteroscedastic=heteroscedastic)
        y_50ci = y_std * 0.6745
        y_95ci = y_std * 1.96

        axs[i].scatter(x, y, label="Data")
        axs[i].plot(x, target_function(x), label="Target function")
        axs[i].fill_between(
            x,
            target_function(x) - y_50ci,
            target_function(x) + y_50ci,
            color="lightblue",
            alpha=0.5,
            label="50% Confidence Interval",
        )
        axs[i].fill_between(
            x,
            target_function(x) - y_95ci,
            target_function(x) + y_95ci,
            color="lightblue",
            alpha=0.2,
            label="95% Confidence Interval",
        )
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].set_title(f"Noise: {'heteroscedastic' if heteroscedastic else 'homoscedastic'}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()
