import numpy as np
import torch
import argparse
import csv
import os
from regressors import *
from metrics import *

from sklearn.model_selection import train_test_split


# ============================================================
# Train/val/test split
# ============================================================

def train_test_split_spatial(coords, Z, Y, state_id, seed):

    idx = np.arange(len(Z))

    idx_train, idx_temp = train_test_split(
        idx,
        test_size=0.3,
        random_state=seed,
        stratify=state_id
    )

    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=2/3,
        random_state=seed
    )

    return {

        "C_train": coords[idx_train],
        "C_val": coords[idx_val],
        "C_test": coords[idx_test],

        "Z_train": Z[idx_train],
        "Z_val": Z[idx_val],
        "Z_test": Z[idx_test],

        "Y_train": Y[idx_train],
        "Y_val": Y[idx_val],
        "Y_test": Y[idx_test],

    }


# ============================================================
# Evaluate
# ============================================================

def evaluate_method(method, split):

    results = {}

    C_train = split["C_train"]
    C_val = split["C_val"]
    C_test = split["C_test"]

    Z_train = split["Z_train"]
    Z_val = split["Z_val"]
    Z_test = split["Z_test"]

    Y_train = split["Y_train"]
    Y_val = split["Y_val"]
    Y_test = split["Y_test"]


    # kNN
    if method in ["knn", "all"]:
        print("Evaluating kNN...")
        ks = [3,5,10,20,50]
        best_score = 1e9

        for k in ks:

            model = KNNRegressor(k=k)
            model.fit(C_train, Y_train)

            score = rmse(Y_val, model.predict(C_val)).mean()

            if score < best_score:

                best_score = score
                best_k = k


        model = KNNRegressor(k=best_k)

        model.fit(
            np.vstack([C_train, C_val]),
            np.vstack([Y_train, Y_val])
        )

        pred = model.predict(C_test)

        results["knn"] = evaluate(Y_test, pred)
        print("kNN done.")

    # IDW
    if method in ["idw", "all"]:
        print("Evaluating IDW...")
        powers = [1,2]
        ks = [10,20,50]

        best_score = 1e9

        for p in powers:
            for k in ks:

                model = IDWRegressor(power=p, k=k)
                model.fit(C_train, Y_train)

                score = rmse(Y_val, model.predict(C_val)).mean()

                if score < best_score:

                    best_score = score
                    best_p = p
                    best_k = k


        model = IDWRegressor(power=best_p, k=best_k)

        model.fit(
            np.vstack([C_train, C_val]),
            np.vstack([Y_train, Y_val])
        )

        pred = model.predict(C_test)

        results["idw"] = evaluate(Y_test, pred)
        print("IDW done.")

    # MLP
    if method in ["mlp", "all"]:
        print("Evaluating MLP...")
        hidden_sizes = [64, 128, 512]

        best_score = 1e9

        for h in hidden_sizes:

            model = MLPRegressorWrapper(hidden_dim=h)

            model.fit(Z_train, Y_train.squeeze())

            score = rmse(
                Y_val.squeeze(),
                model.predict(Z_val)
            ).mean()

            if score < best_score:

                best_score = score
                best_h = h


        model = MLPRegressorWrapper(hidden_dim=best_h)

        model.fit(
            np.vstack([Z_train, Z_val]),
            np.vstack([Y_train, Y_val]).squeeze()
        )

        pred = model.predict(Z_test)

        results["mlp"] = evaluate(
            Y_test,
            pred[:, None]
        )
        print("MLP done.")

    return results


# ============================================================
# Save CSV
# ============================================================

def save_results(results, target, seed, out_file):

    file_exists = os.path.exists(out_file)
    file_empty = (not file_exists) or os.path.getsize(out_file) == 0


    with open(out_file, "a", newline="") as f:

        writer = csv.writer(f)


        # write header only once
        if file_empty:

            writer.writerow(
                ["method", "target", "seed", "MAE", "RMSE", "R2"]
            )


        # append results
        for method in results:

            r = results[method]

            writer.writerow([

                method,
                target,
                seed,

                r["mae"].mean(),
                r["rmse"].mean(),
                r["r2"].mean()

            ])


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser.add_argument("--input", type=str, default=os.path.join(ROOT, "data", "region_graph_with_features_and_targets.npz"), help="Path to dataset .npz file")

    parser.add_argument("--emb", type=str, default=os.path.join(ROOT, "pretrain", "checkpoints", "shared_final_emb_128.pt"), help="Path to embedding .pt file")
    parser.add_argument("--target", choices=["res", "env"], default="res")
    parser.add_argument("--method", choices=["knn", "idw", "mlp", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=os.path.join(ROOT, "Evaluation", "results.csv"), help="Path to output CSV file")
    
    args = parser.parse_args()


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # load data

    region = np.load(args.input)
    coords = np.stack(
        [region["lon"], region["lat"]],
        axis=1
    )
    state_id = region["state_labels"]

    if args.target == "res":
        Y = region["Y_r"][:, None]
    else:
        Y = region["Y_e"][:, None]


    # load embedding

    Z = torch.load(args.emb, weights_only=True).cpu().numpy()

    # split

    split = train_test_split_spatial(
        coords,
        Z,
        Y,
        state_id,
        args.seed
    )

    # evaluate

    results = evaluate_method(
        args.method,
        split
    )


    # save

    save_results(
        results,
        args.target,
        args.seed,
        args.out
    )


    print("Saved to", args.out)


# ============================================================

if __name__ == "__main__":

    main()