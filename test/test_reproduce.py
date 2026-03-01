import numpy as np
import tempfile

from data_generator import generate


def test_reproducibility():

    with tempfile.TemporaryDirectory() as tmp:

        config = {"level": "county"}

        path1 = generate(config, tmp)
        path2 = generate(config, tmp)

        data1 = np.load(path1)
        data2 = np.load(path2)

        assert np.allclose(data1["lon"], data2["lon"])
        assert np.allclose(data1["X_wind"], data2["X_wind"])
        assert np.allclose(data1["X_aq"], data2["X_aq"])
        assert np.allclose(data1["Y_e"], data2["Y_e"])
        assert np.allclose(data1["Y_r"], data2["Y_r"])