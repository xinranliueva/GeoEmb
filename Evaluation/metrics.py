# ============================================================
# metrics.py
# Evaluation metrics for downstream regression
# Supports multi-target regression
# ============================================================

import numpy as np



# ============================================================
# Core metrics
# ============================================================

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred), axis=0)



def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean((y_true - y_pred)**2, axis=0))



def r2(y_true, y_pred):
    """
    R^2 score
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred)**2, axis=0)

    ss_tot = np.sum(
        (y_true - y_true.mean(axis=0))**2,
        axis=0
    )

    return 1 - ss_res / ss_tot



# ============================================================
# Combined evaluation
# ============================================================

def evaluate(y_true, y_pred):
    """
    Returns dict of metrics

    per target and average
    """

    mae_v = mae(y_true, y_pred)
    rmse_v = rmse(y_true, y_pred)
    r2_v = r2(y_true, y_pred)


    results = {

        "mae": mae_v,
        "rmse": rmse_v,
        "r2": r2_v

    }

    return results



# ============================================================
# Pretty print
# ============================================================

def print_results(name, results):
    """
    Prints clean table
    """

    mae_v = results["mae"]
    rmse_v = results["rmse"]
    r2_v = results["r2"]

    print("\n=======================================")
    print(name)
    print("=======================================")

    print(
        f"{'MAE':>12s}"
        f"{'RMSE':>12s}"
        f"{'R2':>12s}"
    )

    print(
        f"{mae_v[0]:12.4f}"
        f"{rmse_v[0]:12.4f}"
        f"{r2_v[0]:12.4f}"
    )

