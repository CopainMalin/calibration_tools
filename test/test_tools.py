import pytest
from pandas import DataFrame, Series
import numpy as np

from src.tools import create_calib_dataset


class TestCalibDataset:
    """Class that regroups all the tests for the create_calib_dataset function."""

    @pytest.fixture
    def calib_dataset(self):
        """
        Creates moking datas to be used in the tests.
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
