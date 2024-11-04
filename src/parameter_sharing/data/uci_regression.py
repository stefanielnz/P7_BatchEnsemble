import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import torch
from scipy.io import arff


def read_arff(filepath) -> pd.DataFrame:
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    df.columns = meta.names()
    return df


def remove_trailing_space(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    lines = [line.replace(" \n", "\n") for line in lines]
    if lines[-1].endswith(" "):
        # Remove trailing space from last line
        lines[-1] = lines[-1].rstrip()
    filepath = filepath.parent / (filepath.stem + "_no_trailing_space" + filepath.suffix)
    with open(filepath, "w") as f:
        f.writelines(lines)
    return filepath


@dataclass
class DataSpec:
    url: str
    label: str
    num_examples: int
    num_input_features: int
    num_output_features: int
    read_params: dict | None
    columns: list[str] | None
    read_function: str = "pd.read_csv"
    preprocess: str | None = None
    file_in_zip: str | None = None
    exclude: list[str] | None = None


data_specs: dict[str, DataSpec] = {
    "boston_housing": DataSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        read_params={"sep": r"\s+", "header": None, "skipinitialspace": True},
        columns=[
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV",
        ],
        label="MEDV",
        num_examples=506,
        num_input_features=13,
        num_output_features=1,
    ),
    "concrete_strength": DataSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        read_function="pd.read_excel",
        read_params={"header": 0},
        columns=[
            "cement",
            "blast_furnace_slag",
            "fly_ash",
            "water",
            "superplasticizer",
            "coarse_aggregate",
            "fine_aggregate",
            "age",
            "concrete_compressive_strength",
        ],
        label="concrete_compressive_strength",
        num_examples=1030,
        num_input_features=8,
        num_output_features=1,
    ),
    "energy_efficiency": DataSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        read_function="pd.read_excel",
        read_params={},
        columns=None,
        label="Y1",
        exclude=["Y2"],
        num_examples=768,
        num_input_features=8,
        num_output_features=1,
    ),
    "kin8nm": DataSpec(
        url="https://www.openml.org/data/download/3626/dataset_2175_kin8nm.arff",
        read_function="read_arff",
        read_params=None,
        columns=None,
        label="'y'",
        num_examples=8192,
        num_input_features=8,
        num_output_features=1,
    ),
    "naval_propulsion": DataSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
        read_params={"header": None, "sep": r"\s+", "skipinitialspace": True},
        columns=[
            "lp",
            "v",
            "GTT",
            "GTn",
            "GGn",
            "Ts",
            "Tp",
            "T48",
            "T1",
            "T2",
            "P48",
            "P1",
            "P2",
            "Pexh",
            "TIC",
            "mf",
            "GTCD",
            "GTTC",
        ],
        file_in_zip="UCI CBM Dataset/data.txt",
        label="GTTC",
        exclude=["GTCD"],
        num_examples=11934,
        num_input_features=16,
        num_output_features=1,
    ),
    "power_plant": DataSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
        read_function="pd.read_excel",
        read_params={"header": 0},
        columns=None,
        file_in_zip="CCPP/Folds5x2_pp.xlsx",
        label="PE",
        num_examples=9568,
        num_input_features=4,
        num_output_features=1,
    ),
    "protein_structure": DataSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        read_params={"header": 0},
        columns=None,
        label="RMSD",
        num_examples=45730,
        num_input_features=9,
        num_output_features=1,
    ),
    "wine_quality_red": DataSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        read_params={"header": 0, "sep": ";"},
        columns=None,
        label="quality",
        num_examples=1599,
        num_input_features=11,
        num_output_features=1,
    ),
    "yacht_hydrodynamics": DataSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        read_params={"header": None, "sep": r"\s+", "skipinitialspace": True},
        columns=[
            "longitudinal_pos",
            "prismatic_coeff",
            "length_disp",
            "beam_draught_ratio",
            "length_beam_ratio",
            "froude_number",
            "resid_resist",
        ],
        preprocess="remove_trailing_space",
        label="resid_resist",
        num_examples=308,
        num_input_features=6,
        num_output_features=1,
    ),
}


def process_dataset(dataset_name: str, data_spec, data_dir: Path) -> Path:
    """Download and process a dataset.

    The folder structure is as follows:
    data_dir/
    ├── raw/
    │   └── [downloaded raw dataset files]
    └── processed/
        └── dataset_name.csv

    Skip downloading and processing if the processed dataset file already exists.

    Returns:
        Path: Path to the processed dataset file.
    """
    raw_data_path = data_dir / "raw"
    raw_data_path.mkdir(exist_ok=True, parents=True)
    processed_data_path = data_dir / "processed"
    processed_data_path.mkdir(exist_ok=True, parents=True)

    processed_dataset_file = processed_data_path / f"{dataset_name}.csv"
    if processed_dataset_file.exists():
        return processed_dataset_file

    raw_dataset_file = raw_data_path / data_spec.url.split("/")[-1]

    if not raw_dataset_file.exists():
        response = requests.get(data_spec.url, timeout=5)
        response.raise_for_status()
        with open(raw_dataset_file, "wb") as file:
            file.write(response.content)

    if data_spec.file_in_zip:
        with zipfile.ZipFile(raw_dataset_file, "r") as zip_ref:
            raw_dataset_file = Path(zip_ref.extract(data_spec.file_in_zip, raw_dataset_file.parent))

    if data_spec.preprocess:
        preprocess_function = globals()[data_spec.preprocess]
        raw_dataset_file = preprocess_function(raw_dataset_file)

    # Dynamically determine the read function (either a global function or a module method)
    if "." in data_spec.read_function:
        module_name, function_name = data_spec.read_function.split(".")
        module = globals().get(module_name)
        if not module:
            raise ValueError(f"Module '{module_name}' not found.")
        read_function = getattr(module, function_name, None)
        if not read_function:
            raise ValueError(f"Function '{function_name}' not found in module '{module_name}'.")
    else:
        read_function = globals().get(data_spec.read_function)
        if not read_function:
            raise ValueError(f"Function '{data_spec.read_function}' not found.")

    dataset_df = read_function(raw_dataset_file, **(data_spec.read_params or {}))
    if data_spec.columns:
        dataset_df.columns = data_spec.columns

    dataset_df.to_csv(processed_dataset_file, index=False, header=True)

    return processed_dataset_file


def get_uci_dataset(name: str, data_dir: str | Path = "data/") -> tuple[torch.Tensor, torch.Tensor]:
    data_dir = Path(data_dir) / "uci" / name
    data_spec = data_specs.get(name, None)
    if data_spec is None:
        raise ValueError(f"Dataset '{name}' not found.")

    data_filepath = process_dataset(name, data_spec, data_dir)
    df = pd.read_csv(data_filepath)
    if data_spec.exclude:
        df = df.drop(columns=data_spec.exclude)

    # Split into data and label
    data = torch.FloatTensor(df.drop(columns=[data_spec.label]).values)
    label = torch.FloatTensor(df[data_spec.label].values)
    # Ensure that label is 2D
    if len(label.shape) == 1:
        label = label.unsqueeze(1)

    if data_spec.num_input_features != data.shape[1]:
        raise ValueError(f"Number of input features in dataset '{name}' does not match expected value.")
    if data_spec.num_output_features != label.shape[1]:
        raise ValueError(f"Number of output features in dataset '{name}' does not match expected value.")

    return data, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec

    # Create the figure and define the gridspec
    fig = plt.figure(figsize=(10, 15))
    gs = gridspec.GridSpec(2, 2, figure=fig)  # 2 rows, 2 columns

    # Summary plots (top row)
    datasets = list(data_specs.keys())
    num_samples = [data_specs[dataset].num_examples for dataset in datasets]
    num_features = [data_specs[dataset].num_input_features for dataset in datasets]

    ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
    ax1.bar(datasets, num_samples)
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Number of Samples per Dataset")

    ax2 = fig.add_subplot(gs[0, 1])  # First row, second column
    ax2.bar(datasets, num_features)
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Number of Features")
    ax2.set_title("Number of Features per Dataset")

    # angle the x-axis labels for the top row
    for ax in [ax1, ax2]:
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right")

    # Violin plot (bottom row, spanning both columns)
    violin_data = []

    for dataset_name in data_specs:
        data_spec = data_specs[dataset_name]
        data_filepath = process_dataset(dataset_name, data_spec, Path("data/") / "uci" / dataset_name)
        df = pd.read_csv(data_filepath)

        for feature_idx, feature in enumerate(df.columns):
            for val in df[feature]:
                violin_data.append([dataset_name, feature_idx, val])

    df = pd.DataFrame(violin_data, columns=["Dataset", "Feature", "Value"])

    ax3 = fig.add_subplot(gs[1, :])  # Second row, span all columns
    sns.boxplot(x="Dataset", y="Value", hue="Feature", data=df, ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.set_title("Distribution of Features for Each Dataset")
    # set y to log
    ax3.set_yscale("log")

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    plt.tight_layout()
    plt.show()
