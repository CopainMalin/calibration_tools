from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from numpy import mean as nmean, std as nstd, arange
from pandas import DataFrame
from seaborn import kdeplot
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay
from typing import List, Dict


def plot_calibration_curve(
    ax: Axes,
    y_test: ArrayLike,
    predictions: Dict[str, ArrayLike],
    colors: list,
    calibration_scores: DataFrame,
) -> None:
    """Plot the calibration curve of the different techniques.

    Args:
        ax (Axes): The axis where to plot the calibration curve.
        y_test (ArrayLike): The true output data.
        predictions (Dict[str, ArrayLike]): The name:predictions dict given by the different techniques.
        colors (list): The colors to use for the plot.
        calibration_scores (DataFrame): The calibration scores of the different techniques.
    """
    for idx, (name, y_prob) in enumerate(predictions.items()):
        prob_true, prob_pred = calibration_curve(
            y_test, y_prob, n_bins=10, strategy="quantile"
        )
        CalibrationDisplay(prob_true, prob_pred, y_prob).plot(
            ax=ax,
            color=colors[idx],
            label=f"{name} | Cross entropy : {calibration_scores.at[name, 'Log loss']:.2f}",
        )
    ax.set_title("Calibration plot", color="black", fontweight="bold")


def plot_uncertainty_distribution(
    ax: Axes, calib_dataset: DataFrame, colors: List[str]
) -> None:
    """Plot the distribution of the uncertainty around the predicted probability.

    Args:
        ax (Axes): The axis where to plot the distribution.
        calib_dataset (DataFrame): The dataset containing the results of the two isotonic regressions fitted in the Venn Arbers calibration process.
        colors (List[str]): The colors to use for the plot.
    """
    mu = nmean(calib_dataset["width"])
    sigma = nstd(calib_dataset["width"])
    kdeplot(
        calib_dataset["width"], fill=True, color=colors[0], label=r"$p_1-p_0$", ax=ax
    )
    ax.axvline(x=mu, color=colors[1], label=f"$\mu$ : {mu:.2f}", alpha=0.7)
    ax.axvline(x=mu + sigma, color=colors[1], linestyle="dashed", alpha=0.7)
    ax.axvline(
        x=mu - sigma,
        color=colors[1],
        linestyle="dashed",
        alpha=0.7,
        label=f"$\sigma$ : {sigma:.2f}",
    )
    ax.set_title(
        "Distribution of uncertainty around the predicted probability (Inductive Venn-Arbers)",
        color="black",
        fontweight="bold",
    )
    ax.set_xlabel("Incertitude ($p_1 - p_0)$")
    ax.legend()


def plot_isotonic_regressions(
    ax: Axes, calib_dataset: DataFrame, colors: List[str]
) -> None:
    """Plot the isotonic regressions fitted in the Venn Arbers calibration process.

    Args:
        ax (Axes): The axis where to plot the isotonic regressions.
        The dataset containing the results of the two isotonic regressions fitted in the Venn Arbers calibration process.
        colors (List[str]): The colors to use for the plot.
    """
    x = arange(calib_dataset.shape[0])
    ax.plot(x, calib_dataset["p0"], color=colors[3], alpha=0.7, label=r"$p_0$")
    ax.plot(x, calib_dataset["p1"], color=colors[3], alpha=0.7, label=r"$p_1$")
    ax.plot(
        x,
        calib_dataset["p"],
        color="black",
        linestyle="--",
        alpha=0.7,
        label=r"$p = \frac{p_1}{1-p_0+p_1}$",
    )
    ax.fill_between(
        x, calib_dataset["p0"], calib_dataset["p1"], color=colors[3], alpha=0.5
    )
    ax.plot(
        x,
        calib_dataset["p1"] - calib_dataset["p0"],
        color=colors[1],
        label=r"$p_1 - p_0$",
    )
    ax.set_xlabel("Sorted samples index")
    ax.set_ylabel("Probability")
    ax.set_title(
        "Isotonic regressions (Inductive Venn-Arbers) results",
        color="black",
        fontweight="bold",
    )
    ax.legend()


def figure_saving(save_path: str = None) -> None:
    """Show the figure if no save path is given, otherwise save it.

    Args:
        save_path (str): The path where to save the figure. Defaults to None.
    """
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")


def plot_calibration(
    y_test: ArrayLike,
    predictions: Dict[str, ArrayLike],
    calib_dataset: DataFrame,
    calib_score: DataFrame,
    save_path: str = None,
    colors: List[str] = None,
) -> None:
    """Plot calibration curve, uncertainty distribution and isotonic regressions.

    Args:
        y_test (ArrayLike): The true output data.
        predictions (Dict[str, ArrayLike]): The name:predictions dict given by the different techniques.
        calib_dataset (DataFrame): The dataset containing the results of the two isotonic regressions fitted in the Venn Arbers calibration process.
        calib_score (DataFrame): The dataset containing the calibration scores of the different techniques.
        save_path (str, optional): The path to save the figure. If None, just show the figure. Defaults to None.
        colors (List[str], optional): The colors to use in the plot. Defaults to None.
    """
    _, axs = plt.subplots(3, 1, figsize=(13, 13))
    plot_calibration_curve(axs[0], y_test, predictions, colors, calib_score)
    plot_uncertainty_distribution(axs[1], calib_dataset, colors)
    plot_isotonic_regressions(axs[2], calib_dataset, colors)

    plt.tight_layout()
    figure_saving(save_path)
