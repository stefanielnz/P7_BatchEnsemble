import typer
import torch
import torch.nn as nn
import seaborn as sns
from parameter_sharing.data.skafte import (
    get_skafte_data,
    target_function,
    scale_function,
)
from parameter_sharing.models.batchensemble import Linear as BELinear
from parameter_sharing.utils.standard_scaler import StandardScaler
import matplotlib.pyplot as plt


# Constants as default parameters to main function
def main(
    ensemble_size: int = 10,
    hidden_size: int = 50,
    alpha_gamma_init: float = 0.5,
    num_steps: int = 250,
    seed: int = 42,
):
    """Main function."""

    torch.manual_seed(seed)

    # Generate data
    x, y = get_skafte_data(num_samples=80, heteroscedastic=True)
    # Split
    test_indices = torch.where(((x >= 5) & (x <= 7)) | ((x >= 11) & (x <= 12.5)))[0]
    train_indices = torch.where(((x < 5) | (x > 7)) & ((x < 11) | (x > 12.5)))[0]
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    # Train models
    fig, axs = plt.subplots(3, 2, figsize=(10, 5))
    for i, model_type in enumerate(["simple", "batchensemble"]):
        model, scaler, train_losses, test_losses = train_model(
            x_train,
            y_train,
            x_test,
            y_test,
            model_type=model_type,
            ensemble_size=ensemble_size,
            hidden_size=hidden_size,
            alpha_gamma_init=alpha_gamma_init,
            num_steps=num_steps,
        )
        plot_results(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            axs[0, i],
            scaler,
            model_type=model_type,
        )
        plot_losses(train_losses, test_losses, axs[1, i])
        plot_parameters(model, axs[2, i])

    plt.tight_layout()
    plt.show()


def train_model(
    x_train,
    y_train,
    x_test,
    y_test,
    model_type="simple",
    ensemble_size=10,
    hidden_size=50,
    alpha_gamma_init=0.5,
    num_steps=300,
):
    """Trains either a simple MLP or a BatchEnsemble MLP."""

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if model_type == "simple":
        model = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    elif model_type == "batchensemble":
        model = nn.Sequential(
            BELinear(
                1,
                hidden_size,
                ensemble_size=ensemble_size,
                alpha_init=alpha_gamma_init,
                gamma_init=alpha_gamma_init,
            ),
            nn.ReLU(),
            BELinear(
                hidden_size,
                hidden_size,
                ensemble_size=ensemble_size,
                alpha_init=alpha_gamma_init,
                gamma_init=alpha_gamma_init,
            ),
            nn.ReLU(),
            BELinear(
                hidden_size,
                1,
                ensemble_size=ensemble_size,
                alpha_init=alpha_gamma_init,
                gamma_init=alpha_gamma_init,
            ),
        )
    else:
        raise ValueError("Invalid model_type. Choose 'simple' or 'batchensemble'")

    lr = 0.003
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses = []
    test_losses = []
    for step in range(num_steps - 1):
        # Draw 10 samples
        indices = torch.randperm(x_train.size(0))[: 3 * ensemble_size]
        x_train_batch = x_train[indices]
        y_train_batch = y_train[indices]
        optimizer.zero_grad()
        y_pred = model(x_train_batch)
        mse = loss_fn(y_pred, y_train_batch)
        loss = mse
        loss.backward()
        optimizer.step()
        train_losses.append(mse.item())

        with torch.no_grad():
            if model_type == "batchensemble":
                y_pred = model(x_test.repeat(ensemble_size, 1)).view(ensemble_size, x_test.size(0), 1).mean(dim=0)
            else:
                y_pred = model(x_test)
            test_loss = loss_fn(y_pred, y_test).item()
            test_losses.append(test_loss)

    return model, scaler, train_losses, test_losses


@torch.no_grad()
def plot_results(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    ax,
    scaler,
    model_type="simple",
    x_range=(2.5, 12.5),
):
    """Plots the results of the model."""

    x_linspace = torch.linspace(x_range[0], x_range[1], 100).unsqueeze(1)
    x_linspace_scaled = scaler.transform(x_linspace)

    if model_type == "batchensemble":
        y_pred = model(x_linspace_scaled.repeat(model[0].ensemble_size, 1)).view(
            model[0].ensemble_size, x_linspace.size(0), 1
        )
    else:
        y_pred = model(x_linspace_scaled)

    x = x_linspace.squeeze()
    ax.fill_between(
        x,
        target_function(x) - scale_function(x, heteroscedastic=True),
        target_function(x) + scale_function(x, heteroscedastic=True),
        color="gray",
        alpha=0.2,
        label="Data noise std",
    )

    ax.scatter(x_train, y_train, label="Train data")
    ax.scatter(x_test, y_test, label="Test data")
    if model_type == "batchensemble":
        ax.plot(
            x_linspace,
            y_pred.mean(dim=0),
            label=f"{model_type.capitalize()} MLP prediction (mean)",
            color="blue",
        )
        for i in range(model[0].ensemble_size):
            ax.plot(x_linspace, y_pred[i], alpha=0.5, color="lightblue")
    else:
        ax.plot(x_linspace, y_pred, label=f"{model_type.capitalize()} MLP prediction", color="blue")
    ax.plot(x_linspace, target_function(x_linspace), label="Target function", color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{model_type.capitalize()} MLP")
    ax.legend()


def plot_losses(train_losses, test_losses, ax):
    """Plots the training and testing losses."""
    ax.plot(train_losses, label="Train loss")
    ax.plot(test_losses, label="Test loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Testing Loss")
    ax.legend()


@torch.no_grad()
def plot_parameters(model, ax):
    """Plots the parameters of the model."""
    # Flatten all parameters, then plot in descending order
    params = []
    tags = []
    for name, param in model.named_parameters():
        if "bias" in name:
            tag = "bias"
        elif "alpha" in name:
            tag = "alpha"
        elif "gamma" in name:
            tag = "gamma"
        elif "weight" in name:
            tag = "weight"
        else:
            tag = "unknown"
        params.append(param.flatten())
        tags += [tag] * param.numel()
    params = torch.cat(params)
    sorted_indices = torch.argsort(params, descending=True)
    tags = [tags[i] for i in sorted_indices]
    params = params[sorted_indices]

    tag_counts = {}
    for tag in tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    tags_w_counts = [f"{tag} (n={tag_counts[tag]})" for tag in tags]

    sns.histplot(x=params.tolist(), hue=tags_w_counts, ax=ax)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Count")
    ax.set_title("Parameter distribution")


if __name__ == "__main__":
    typer.run(main)
