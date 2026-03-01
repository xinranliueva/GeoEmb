import numpy as np
import tempfile
import os

from data_generator import generate


def test_dataset_generation():

    with tempfile.TemporaryDirectory() as tmp:

        config = {"level": "county"}

        path = generate(config, tmp)

        assert os.path.exists(path)

        data = np.load(path)

        assert "lon" in data
        assert "lat" in data
        assert "edges" in data
        assert "X_wind" in data
        assert "X_aq" in data
        assert "Y_r" in data
        assert "Y_e" in data

        N = len(data["lon"])

        assert data["X_wind"].shape[0] == N
        assert data["X_aq"].shape[0] == N
        assert data["Y_r"].shape[0] == N