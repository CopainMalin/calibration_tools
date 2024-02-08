from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from sklearn.base import ClassifierMixin
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict, List, Tuple
from venn_abers import VennAbersCalibrator

from src.VennABERS.VennABERS import ScoresToMultiProbs
from src.plotting_tools import plot_calibration


## Main function
def calibration_evaluation(
    estimator: ClassifierMixin,
    X_train: ArrayLike,
    X_calib: ArrayLike,
    X_test: ArrayLike,
    y_train: ArrayLike,
    y_calib: ArrayLike,
    y_test: ArrayLike,
    colors: List[str] = ["C0", "C1", "C2", "C3"],
    save_path: str = None,
    cross_va: bool = True,
) -> None:
    """Performs the calibration evaluation of a classifier and plots the results.

    Args:
        estimator (ClassifierMixin): The classifier to evaluate.
        X_train (ArrayLike): The train input data.
        X_calib (ArrayLike): The calibration input data.
        X_test (ArrayLike): The test input data.
        y_train (ArrayLike): The train output data.
        y_calib (ArrayLike): The calibration output data.
        y_test (ArrayLike): The test output data.
        colors (List[str], optional): The colors used in the plots. Defaults to ["C0", "C1", "C2", "C3"].
        save_path (str, optional): The path where to save the plot. If set to None, the plot will just be showed. Defaults to None.
        cross_va (bool, optional): Whether to perform cross Venn Arbers calibration or not (could be time consuming). Defaults to True.

    Returns:
        None
    """
    predictions, p_cal, p_test = probabilistic_predictions(
        estimator, X_train, X_calib, X_test, y_train, y_calib, cross_va
    )
    calib_score = compute_calibration_score(y_test, predictions)
    calib_dataset = create_calib_dataset(p_cal, y_calib, p_test, y_test)

    plot_calibration(
        y_test=y_test,
        predictions=predictions,
        calib_dataset=calib_dataset,
        calib_score=calib_score,
        colors=colors,
        save_path=save_path,
    )
    return calib_score


## Create calibration dataset (results of the two isotonic regressions)
def create_calib_dataframe(
    y_test: ArrayLike, p0: ArrayLike, p1: ArrayLike
) -> DataFrame:
    """Create the calibration dataset containing the results of the two isotonic regressions fitted in the Venn Arbers calibration process.
    More details here : https://alrw.net/articles/13.pdf.

    Args:
        y_test (ArrayLike): The true output data.
        p0 (ArrayLike): The lower probabilities bound.
        p1 (ArrayLike): The upper probabilities bound.

    Returns:
        DataFrame: The calibration dataset.
    """
    calib_dataset = y_test.to_frame() if type(y_test) == Series else DataFrame(y_test)
    calib_dataset.columns = ["y_true"]
    calib_dataset["p0"] = p0
    calib_dataset["p1"] = p1
    calib_dataset["p"] = p1 / (1 - p0 + p1)
    calib_dataset["width"] = p1 - p0
    return calib_dataset


def sort_and_reset_index(df: DataFrame) -> DataFrame:
    """Sort the calibration dataset by the p column and reset the index.

    Args:
        df (DataFrame): The dataframe to be sorted and reset.

    Returns:
        DataFrame: The sorted and reset dataframe.
    """
    df = df.sort_values(by=["p"])
    df = df.reset_index(drop=True)
    return df


def create_calib_dataset(
    p_cal: ArrayLike, y_calib: ArrayLike, p_test: ArrayLike, y_test: ArrayLike
) -> DataFrame:
    """Create the calibration dataset containing the results of the two isotonic regressions fitted in the Venn Arbers calibration process, and sort it by the probabilty column.

    Args:
        p_cal (ArrayLike): The predicted probabilities on the calibration set.
        y_calib (ArrayLike): The true calibration output data.
        p_test (ArrayLike): The predicted probabilities on the test set.
        y_test (ArrayLike): The true test output data.

    Returns:
        DataFrame: The sorted calibration dataset.
    """
    p0, p1 = ScoresToMultiProbs(zip(p_cal, y_calib), p_test)
    calib_dataset = create_calib_dataframe(y_test, p0, p1)
    calib_dataset = sort_and_reset_index(calib_dataset)
    return calib_dataset


## Calibration score
def compute_calibration_score(
    y_test: ArrayLike, predictions: Dict[str, ArrayLike]
) -> DataFrame:
    """Compute the Brier score and the log loss for each set of predictions.

    Args:
        y_test (ArrayLike): The true class of the test set.
        predictions (Dict[str, ArrayLike]): The predicted probabilities for each set of predictions.

    Returns:
        DataFrame: The Brier score and the log loss for each set of predictions.
    """
    res = DataFrame(columns=["Brier score", "Log loss"], index=predictions.keys())
    for name, y_prob in predictions.items():
        res.at[name, "Brier score"] = brier_score_loss(y_test, y_prob)
        res.at[name, "Log loss"] = log_loss(y_test, y_prob)
    return res


## Venn Arbers calibration
def inductive_venn_abers(
    p_cal: ArrayLike, y_calib: ArrayLike, p_test: ArrayLike
) -> ArrayLike:
    """Performs inductive Venn Abers calibration.

    Args:
        p_cal (ArrayLike): predicted probabilities of the calibration set.
        y_calib (ArrayLike): true class of the calibration set.
        p_test (ArrayLike): predicted probabilities of the test set.

    Returns:
        ArrayLike: The calibrated probabilities of the test set.
    """
    vac = VennAbersCalibrator()
    return vac.predict_proba(p_cal=p_cal, y_cal=y_calib, p_test=p_test)[:, 1]


def cross_venn_abers(
    estimator: ClassifierMixin,
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    n_splits: int = 5,
) -> ArrayLike:
    """Performs cross-validated Venn Abers calibration.

    Args:
        estimator (ClassifierMixin): The classifier to be calibrated.
        X_train (ArrayLike): The train input data.
        y_train (ArrayLike): The train output data.
        X_test (ArrayLike): The test input data.
        n_splits (int, optional): The number of splits for the cross-validation. Defaults to 5.

    Returns:
        ArrayLike: The calibrated probabilities of the test set.
    """
    cvac = VennAbersCalibrator(estimator=estimator, inductive=False, n_splits=n_splits)
    cvac.fit(X_train, y_train)
    return cvac.predict_proba(X_test)[:, 1]


def probabilistic_predictions(
    estimator: ClassifierMixin,
    X_train: ArrayLike,
    X_calib: ArrayLike,
    X_test: ArrayLike,
    y_train: ArrayLike,
    y_calib: ArrayLike,
    cross_va: bool = True,
) -> Tuple[Dict[str, ArrayLike], ArrayLike, ArrayLike]:
    """Performs probabilistic predictions and calibrations.

    Args:
        estimator (ClassifierMixin): The classifier to be calibrated.
        X_train (ArrayLike): The train input data.
        X_calib (ArrayLike): The calibration input data.
        X_test (ArrayLike): The test input data.
        y_train (ArrayLike): The train output data.
        y_calib (ArrayLike): The calibration output data.
        cross_va (bool, optional): Whether to perform cross Venn Arbers calibration or not (could be time consuming). Defaults to True.

    Returns:
        Tuple[Dict[str, ArrayLike], ArrayLike, ArrayLike]: The calibrated predictions, the non-calibrated probabilities on the calibration set and the non-calibrated probabilities on the test set.
    """

    p_cal = estimator.predict_proba(X_calib)
    p_test = estimator.predict_proba(X_test)

    predictions = {"No calibration": p_test[:, 1]}
    predictions["Inductive Venn-ABERS"] = inductive_venn_abers(p_cal, y_calib, p_test)
    if cross_va:
        predictions["Cross Venn-ABERS"] = cross_venn_abers(
            estimator, X_train, y_train, X_test
        )

    return predictions, p_cal[:, 1], p_test[:, 1]
