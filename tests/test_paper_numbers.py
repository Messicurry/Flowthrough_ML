"""Published values from Table 4 and Section 3.7. Takes a couple of minutes."""
import pytest

from flowthrough.data import REMOVAL, STUDY, load
from flowthrough.evaluate import (cross_validate, leave_one_study_out, offset_share,
                                  one_point_calibration, scores, summarize)


@pytest.fixture(scope="module")
def dataset():
    return load()


@pytest.mark.parametrize("route, r2, sd", [("R1", 0.766, 0.019), ("R4", 0.783, 0.016), ("R6", 0.818, 0.019)])
def test_cross_validation(dataset, route, r2, sd):
    table = summarize(cross_validate(*dataset, route))
    assert round(table.loc[route, ("R2", "mean")], 3) == r2
    assert round(table.loc[route, ("R2", "std")], 3) == sd


def test_leave_one_study_out(dataset):
    data, dictionary = dataset
    y = data[REMOVAL].to_numpy(float)
    studies = data[STUDY].to_numpy()
    pred = leave_one_study_out(data, dictionary, "R6")
    assert round(scores(y, pred)["R2"], 3) == -0.104
    assert round(offset_share(y, pred, studies), 3) == 0.822
    assert round(scores(*one_point_calibration(y, pred, studies))["R2"], 3) == 0.474
