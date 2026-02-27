import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.ndimage import gaussian_filter


def generate_wind(
    n=200,
    # harmonic generator
    lmax_base=15,
    decay_base=2.0,
    seed_base=1,
    lmax_turb=60,
    decay_turb=1.2,
    seed_turb=42,
    # land + coast roughness
    land_sigma=8,
    land_mode="wrap",
    # blending
    trade_wind_strength=2.0,
    base_weight=0.8,
    turb_weight=0.3,
    storm_track_center_deg=45.0,
    storm_track_width=0.15,
    # coastal jitter
    seed_jitter=7,
    jitter_std=0.08,
    # divergent “leakage”
    convergence_strength=0.2,
    # friction over land
    land_friction=0.4,
    # output controls
    return_psi=False,
    save_path=None,
    # plotting (optional)
    plot=False,
    plot_path="wind_plot.png",
    step=3,
    density=2,
):
    """
    Synthetic wind generator over a CONUS lat/lon grid.

    Notes (intended design):
    - Builds a smooth random streamfunction via spherical harmonics evaluated on a
      CONUS patch that is reparameterized to global coordinates.
    - Adds a mid-latitude turbulence envelope and a zonal background component.
    - Uses a smoothed land mask to modulate friction and define coastal roughness.
    - Optionally saves outputs to NPZ (lon/lat in degrees; u/v/speed in model units).
    """

    def smooth_field(n=200, lmax=20, decay_power=2.5, seed=1):
        rng = np.random.default_rng(seed)

        # --- CONUS grid (actual output grid) ---
        lat = np.deg2rad(np.linspace(25, 49, n))
        lon = np.deg2rad(np.linspace(-125, -66, n))
        Lon, Lat = np.meshgrid(lon, lat)

        phi = (Lon - lon.min()) / (lon.max() - lon.min()) * (2 * np.pi) - np.pi
        Lat_global = (Lat - lat.min()) / (lat.max() - lat.min()) * np.pi - np.pi / 2
        theta = np.pi / 2 - Lat_global

        f = np.zeros_like(theta, dtype=complex)

        for l in range(1, lmax + 1):
            amp = 1 / (1 + l) ** decay_power
            for m in range(-l, l + 1):
                coeff = amp * (rng.normal() + 1j * rng.normal())
                f += coeff * sph_harm(m, l, phi, theta)

        f = np.real(f)
        f = (f - f.mean()) / f.std()

        return Lat, Lon, f

    # ============================================================
    # 1) Base + turbulence fields
    # ============================================================
    Lat, Lon, psi_base = smooth_field(n=n, lmax=lmax_base, decay_power=decay_base, seed=seed_base)
    _, _, psi_turb = smooth_field(n=n, lmax=lmax_turb, decay_power=decay_turb, seed=seed_turb)

    # ============================================================
    # 2) Land mask + coastal roughness proxy
    # ============================================================
    lat_deg = np.rad2deg(Lat)
    lon_deg = np.rad2deg(Lon)

    land = (
        ((np.abs(lon_deg + 60) < 35) & (lat_deg > -10) & (lat_deg < 60)) |
        ((np.abs(lon_deg - 20) < 45) & (lat_deg > -35) & (lat_deg < 70)) |
        ((np.abs(lon_deg - 110) < 35) & (lat_deg > -10) & (lat_deg < 55))
    )

    land_smooth = gaussian_filter(land.astype(float), sigma=land_sigma, mode=land_mode)

    dM_dlat, dM_dlon = np.gradient(land_smooth)
    coast_roughness = np.sqrt(dM_dlat**2 + dM_dlon**2)

    # ============================================================
    # 3) Selective blending / physical biasing
    # ============================================================
    storm_track_envelope = np.exp(-(np.abs(Lat) - np.deg2rad(storm_track_center_deg))**2 / storm_track_width)

    psi_trades = trade_wind_strength * np.sin(Lat)

    psi_physical = (
        (psi_base * base_weight) +
        (psi_trades) +
        (psi_turb * turb_weight * storm_track_envelope)
    )

    rng = np.random.default_rng(seed_jitter)
    psi_physical += rng.normal(0, jitter_std, Lat.shape) * coast_roughness

    # ============================================================
    # 4) Wind from (mostly) rotational field + small divergent component
    # ============================================================
    dlat = Lat[1, 0] - Lat[0, 0]
    dlon = Lon[0, 1] - Lon[0, 0]
    coslat_safe = np.clip(np.cos(Lat), 0.1, 1.0)

    dpsi_dlat, dpsi_dlon = np.gradient(psi_physical, dlat, dlon)

    u_rot = -dpsi_dlat
    v_rot = dpsi_dlon / coslat_safe

    # small divergent “leakage”
    u_div = 0.0
    v_div = -convergence_strength * np.sin(Lat)

    # friction over land
    friction = (1.0 - land_friction * land_smooth)

    u = (u_rot + u_div) * friction
    v = (v_rot + v_div) * friction
    speed = np.sqrt(u**2 + v**2)

    # ============================================================
    # 5) Save (optional) + return
    # ============================================================
    if save_path is not None:
        np.savez(save_path, lon=lon_deg, lat=lat_deg, u=u, v=v, speed=speed)

    # ============================================================
    # 6) Plot (optional) + save plot
    # ============================================================
    if plot:

        plt.figure(figsize=(10,4))

        im = plt.imshow(
            speed,
            origin="lower",
            extent=[lon_deg.min(), lon_deg.max(), lat_deg.min(), lat_deg.max()],
            aspect="auto",
            cmap="plasma",
            vmin=np.percentile(speed,5),
            vmax=np.percentile(speed,95),
        )

        plt.streamplot(
            lon_deg[0, ::step],          # x grid (1D, increasing)
            lat_deg[::step, 0],          # y grid (1D, increasing)
            u[::step, ::step],
            v[::step, ::step],
            color="white",
            density=density,
        )

        # optional coastline-ish contour 
        # plt.contour(lon_deg, lat_deg, land_smooth, levels=[0.5], colors="white", linewidths=1.0)

        plt.colorbar(im, label="Wind speed")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.xlim(lon_deg.min(), lon_deg.max())
        plt.ylim(lat_deg.min(), lat_deg.max())

        plt.title("Wind Intensity and Streamlines (CONUS)")

        if plot_path is not None:
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")

        plt.show()

    out = {
        "Lat": Lat,
        "Lon": Lon,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "u": u,
        "v": v,
        "speed": speed,
        "land_smooth": land_smooth,
        "coast_roughness": coast_roughness,
    }
    if return_psi:
        out["psi_base"] = psi_base
        out["psi_turb"] = psi_turb
        out["psi_physical"] = psi_physical

    return out