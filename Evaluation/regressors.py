# ============================================================
# regressors.py
# Simple downstream regressors for embedding evaluation
# Includes:
#   - Ridge regression
#   - MLP regressor
#   - kNN regression
#   - IDW regression
# ============================================================

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# Base wrapper (handles scaling safely)
# ============================================================

class BaseRegressor:

    def __init__(self, scale=True):

        self.scale = scale
        self.model = None


    def fit(self, X, Y):

        if self.scale:

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self.model.fit(X, Y)


    def predict(self, X):

        if self.scale:

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        return self.model.predict(X)



# ============================================================
# Ridge Regression
# ============================================================

class RidgeRegressor(BaseRegressor):

    def __init__(self, alpha=1.0, scale=True):

        super().__init__(scale)

        self.model = Ridge(
            alpha=alpha,
            random_state=0
        )



# ============================================================
# MLP Regressor
# ============================================================

class MLPRegressorWrapper(BaseRegressor):

    def __init__(
        self,
        hidden_dim=128,
        max_iter=2000,
        scale=True
    ):

        super().__init__(scale)

        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,hidden_dim,hidden_dim,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=0
        )



# ============================================================
# kNN Regressor (for coordinate baseline)
# ============================================================

class KNNRegressor(BaseRegressor):

    def __init__(self, k=10, weights="distance", scale=True):

        super().__init__(scale)

        self.model = KNeighborsRegressor(
            n_neighbors=k,
            weights=weights,
            metric="haversine"
        )


    def _prepare(self, X):
        """
        Convert (lon, lat) degrees -> (lat, lon) radians
        """

        X = np.asarray(X)

        X = X[:, [1, 0]]      # swap lon,lat -> lat,lon

        X = np.radians(X)

        return X


    def fit(self, X, Y):

        X = self._prepare(X)

        self.model.fit(X, Y)


    def predict(self, X):

        X = self._prepare(X)

        return self.model.predict(X)



# ============================================================
# IDW Regressor
# ============================================================

import numpy as np


EARTH_RADIUS_KM = 6371.0


def haversine_distance(X1, X2):
    """
    X1, X2 in radians, (lat, lon)
    """

    lat1 = X1[:, 0][:, None]
    lon1 = X1[:, 1][:, None]

    lat2 = X2[:, 0][None, :]
    lon2 = X2[:, 1][None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat/2)**2 +
        np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    )

    c = 2*np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c



class IDWRegressor:


    def __init__(self, power=2, k=None, eps=1e-8):

        self.power = power
        self.k = k
        self.eps = eps

        self.X_train = None
        self.Y_train = None


    def _prepare(self, X):

        X = np.asarray(X)

        X = X[:, [1, 0]]   # swap lon,lat -> lat,lon

        X = np.radians(X)

        return X


    def fit(self, X, Y):

        self.X_train = self._prepare(X)

        self.Y_train = Y


    def predict(self, X):

        X = self._prepare(X)

        D = haversine_distance(X, self.X_train)

        preds = []

        for i in range(len(X)):

            dists = D[i]

            if self.k is not None:

                idx = np.argsort(dists)[:self.k]

                dists = dists[idx]
                Y = self.Y_train[idx]

            else:

                Y = self.Y_train


            weights = 1.0 / (dists + self.eps)**self.power

            weights /= weights.sum()

            pred = weights @ Y

            preds.append(pred)


        return np.vstack(preds)