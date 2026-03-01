import numpy as np

from eval import train_test_split_spatial


def test_split():

    N = 200

    coords = np.random.randn(N,2)

    Z = np.random.randn(N,16)

    Y = np.random.randn(N,1)

    state = np.random.randint(0,5,N)

    split = train_test_split_spatial(
        coords, Z, Y, state, seed=0
    )

    assert len(split["Z_train"]) > 0
    assert len(split["Z_test"]) > 0

    assert split["Z_train"].shape[1] == 16