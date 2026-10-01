import numpy as np
import pytest

from flowthrough.data import HRT, LN_K, REMOVAL, STUDY, feature_columns, load
from flowthrough.models import decode


@pytest.fixture(scope="module")
def dataset():
    return load()


def test_dataset_size(dataset):
    data, _ = dataset
    assert len(data) == 222
    assert data[STUDY].nunique() == 40
    assert data["pollutant_name_std"].nunique() == 29


def test_feature_sets(dataset):
    _, dictionary = dataset
    assert [len(cols) for cols in feature_columns(dictionary, "raw")] == [36, 10]
    assert [len(cols) for cols in feature_columns(dictionary, "physics")] == [40, 10]


def test_decoder_gives_back_removal(dataset):
    # removal of 100 % is stored as 99.99 % before taking the log
    data, _ = dataset
    removal = decode(data[LN_K].to_numpy(float), data[HRT].to_numpy(float))
    assert np.max(np.abs(removal - data[REMOVAL].to_numpy(float))) < 0.02
