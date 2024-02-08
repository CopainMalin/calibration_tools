from argparse import ArgumentParser
import numpy as np
from numpy.typing import ArrayLike
import pytest
from pandas import DataFrame
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple
from venn_abers import VennAbersCalibrator

from src.tools import (
    create_calib_dataset,
    compute_calibration_score,
    probabilistic_predictions,
    inductive_venn_abers,
    cross_venn_abers,
)

# note : the tests are splitted in classes, one for each main function of the tools.py file.


class TestCalibDataset:
    """Class that regroups all the tests for the create_calib_dataset function."""

    @pytest.fixture
    def calib_dataset(self):
        """
        Create moking datas to be used in the tests.
        """
        p_cal = np.random.rand(10)
        y_calib = np.random.randint(0, 2, 10)
        p_test = np.random.rand(10)
        y_test = np.random.randint(0, 2, 10)

        return create_calib_dataset(p_cal, y_calib, p_test, y_test)

    # Tests
    def test_columns(self, calib_dataset: DataFrame) -> None:
        """
        Test that the DataFrame has the expected columns.
        """
        assert set(calib_dataset.columns) == {"y_true", "p0", "p1", "p", "width"}

    def test_p0_leq_p1(self, calib_dataset: DataFrame) -> None:
        """
        Test that each value in the 'p0' column is less than or equal to the corresponding value in the 'p1' column.
        """
        assert all(calib_dataset["p0"] <= calib_dataset["p1"])

    def test_sorted_by_p(self, calib_dataset: DataFrame) -> None:
        """
        Test that the values in the 'p' column are sorted in ascending order.
        """
        assert all(calib_dataset["p"].sort_values() == calib_dataset["p"])

    def test_index_reset(self, calib_dataset: DataFrame) -> None:
        """
        Test that the DataFrame's index has been reset to the default integer index.
        """
        assert calib_dataset.index.tolist() == list(range(len(calib_dataset)))


class TestCalibrationScore:
    @pytest.fixture
    def moke_datas(self) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        Generate some mock data.
        """
        y_test = np.random.randint(0, 2, 10)
        predictions = {
            "model1": np.random.rand(10),
            "model2": np.random.rand(10),
            "model3": np.random.rand(10),
        }
        return y_test, predictions

    @pytest.fixture
    def calib_scores(
        self, moke_datas: Tuple[ArrayLike, Dict[str, ArrayLike]]
    ) -> DataFrame:
        """
        Create the calibration scores DataFrame based on the mock datas.
        """
        y_test, predictions = moke_datas
        return compute_calibration_score(y_test, predictions)

    def test_index(
        self,
        calib_scores: DataFrame,
        moke_datas: Tuple[ArrayLike, Dict[str, ArrayLike]],
    ):
        """
        Test that the index of the calibration scores DataFrame is the same as the keys of the predictions dictionary (i.e. the names of the models).
        """
        _, predictions = moke_datas
        assert set(calib_scores.index) == set(predictions.keys())

    def test_columns(
        self,
        calib_scores: DataFrame,
    ):
        """
        Test that the columns of the calibration scores DataFrame (i.e the name of the scores) are the expected ones.
        """
        assert set(calib_scores.columns) == set({"Brier score", "Log loss"})

    def test_values(
        self,
        calib_scores: DataFrame,
        moke_datas: Tuple[ArrayLike, Dict[str, ArrayLike]],
    ):
        """
        Test that the values of the calibration scores DataFrame are the expected ones based on the scikit-learn implementation.
        """
        y_test, predictions = moke_datas
        # Check the values
        for name, y_prob in predictions.items():
            assert np.isclose(
                calib_scores.at[name, "Brier score"], brier_score_loss(y_test, y_prob)
            )
            assert np.isclose(
                calib_scores.at[name, "Log loss"], log_loss(y_test, y_prob)
            )


class TestProbabilisticPrediction:
    @pytest.fixture
    def sets(self):
        """
        Create moke train & calib sets.
        """
        return {
            "X_train": np.random.rand(100, 2),
            "X_calib": np.random.rand(100, 2),
            "X_test": np.random.rand(100, 2),
            "y_train": np.random.randint(0, 2, 100),
            "y_calib": np.random.randint(0, 2, 100),
        }

    @pytest.fixture
    def estimator(self, sets: Dict[str, ArrayLike]) -> LogisticRegression:
        """
        Create a logistic regression model and fit it on the training set.
        """
        estimator = LogisticRegression()
        estimator.fit(sets["X_train"], sets["y_train"])
        return estimator

    @pytest.fixture
    def pred_results(
        self, estimator: LogisticRegression, sets: Dict[str, ArrayLike]
    ) -> Tuple[Dict[str, ArrayLike], ArrayLike, ArrayLike]:
        """
        Compute the probabilistic predictions and the predictions of the estimator on the test set.
        """
        result, p_cal, p_test = probabilistic_predictions(
            estimator,
            X_train=sets["X_train"],
            X_calib=sets["X_calib"],
            X_test=sets["X_test"],
            y_train=sets["y_train"],
            y_calib=sets["y_calib"],
            cross_va=True,
        )
        return result, p_cal, p_test

    def test_keys(
        self, pred_results: Tuple[Dict[str, ArrayLike], ArrayLike, ArrayLike]
    ) -> None:
        """
        Test that the keys of the result dictionary are the expected ones.
        """
        result, _, _ = pred_results
        assert set(result.keys()) == {
            "No calibration",
            "Inductive Venn-ABERS",
            "Cross Venn-ABERS",
        }

    def test_results(
        self,
        sets: Dict[str, ArrayLike],
        estimator: LogisticRegression,
        pred_results: Tuple[Dict[str, ArrayLike], ArrayLike, ArrayLike],
    ):
        """
        Test that the results are the expected ones.
        """
        result, p_cal, p_test = pred_results
        assert np.allclose(
            result["No calibration"], estimator.predict_proba(sets["X_test"])[:, 1]
        )
        assert np.allclose(p_cal, estimator.predict_proba(sets["X_calib"])[:, 1])
        assert np.allclose(p_test, estimator.predict_proba(sets["X_test"])[:, 1])
