import os
import argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from wind import generate_wind
from pollution import generate_pollution
from region import generate_regional_data
from target import generate_regression_targets


# ============================================================
# Main dataset generator
# ============================================================

def generate(config: dict, out_dir: str) -> str:
    
    """
        Writes dataset artifacts and returns the dataset path. 
        Seeds already fixed inside subgenerators for reproducibility.
    """

    os.makedirs(out_dir, exist_ok=True)


    # ============================================================
    # 1. Generate region graph
    # ============================================================

    region = generate_regional_data(
        level=config.get("level", "postal")
    )

    lon = region["coords"][:, 0]
    lat = region["coords"][:, 1]
    state_labels = region["state_labels"]
    edges = region["edges"]

    np.savez(
        os.path.join(out_dir, "region_data.npz"),
        lon=lon,
        lat=lat,
        state_labels=state_labels,
        edges=edges,
    )


    # ============================================================
    # 2. Generate wind
    # ============================================================

    wind = generate_wind(
        save_path=os.path.join(out_dir, "wind_data.npz")
    )

    lon_wind = wind["lon_deg"]
    lat_wind = wind["lat_deg"]
    u = wind["u"]
    v = wind["v"]
    speed = wind["speed"]


    # ============================================================
    # 3. Generate pollution
    # ============================================================

    aq = generate_pollution(
        wind_npz_path=os.path.join(out_dir, "wind_data.npz"),
        save_path=os.path.join(out_dir, "aq_data.npz"),
    )

    lon_aq = aq["lon_deg"]
    lat_aq = aq["lat_deg"]
    pollution_con = aq["aq"]


    # ============================================================
    # 4. Interpolate wind
    # ============================================================

    lon1d = lon_wind[0, :]
    lat1d = lat_wind[:, 0]

    interp_u = RegularGridInterpolator((lat1d, lon1d), u, bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((lat1d, lon1d), v, bounds_error=False, fill_value=None)

    region_points = np.column_stack([lat, lon])

    u_interp = interp_u(region_points)
    v_interp = interp_v(region_points)

    wind_speed = np.sqrt(u_interp**2 + v_interp**2)

    X_wind = np.column_stack([u_interp, v_interp, wind_speed])


   # ============================================================
    # 5. Interpolate air quality (exact grid version)
    # ============================================================

    lon_unique = np.unique(lon_aq)
    lat_unique = np.unique(lat_aq)

    aq_grid = np.full((len(lat_unique), len(lon_unique)), np.nan)
    lat_idx = np.searchsorted(lat_unique, lat_aq)
    lon_idx = np.searchsorted(lon_unique, lon_aq)
    aq_grid[lat_idx, lon_idx] = pollution_con

    interp_aq = RegularGridInterpolator(
        (lat_unique, lon_unique),
        aq_grid,
        bounds_error=False,
        fill_value=None
    )

    aq_interp = interp_aq(np.column_stack([lat, lon]))

    X_aq = aq_interp.reshape(-1,1)


    # ============================================================
    # 6. Targets (uses default seed inside function)
    # ============================================================

    coords = np.column_stack([lon, lat])

    Y = generate_regression_targets(
        coords,
        X_wind,
        X_aq,
    )

    Y_e = Y[:, 0]
    Y_r = Y[:, 1]


    # ============================================================
    # 7. Save final dataset
    # ============================================================

    final_path = os.path.join(
        out_dir,
        "region_graph_with_features_and_targets.npz"
    )

    np.savez(
        final_path,
        lon=lon,
        lat=lat,
        state_labels=state_labels,
        edges=edges,
        X_wind=X_wind,
        X_aq=X_aq,
        Y_r=Y_r,
        Y_e=Y_e,
    )

    print("Saved:", final_path)

    return final_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate synthetic multimodal spatial dataset")
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "data"), help="Output directory")
    parser.add_argument("--level", type=str, default="postal", choices=["postal", "county"], help="Sampling level")
    
    args = parser.parse_args()

    config = {"level": args.level}

    dataset_path = generate(config=config, out_dir=args.out_dir)
    print("\nDataset generated at:", dataset_path)
