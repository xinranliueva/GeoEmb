import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator


def generate_pollution(
    n=800,
    seed=0,
    H=2000,
    # CONUS bounds (degrees)
    lon_bounds=(-125, -66),
    lat_bounds=(25, 49),
    # decay scale (meters) and earth radius (meters)
    decay_km=60.0,
    R_earth=6371000.0,
    # wind penalty
    wind_npz_path="wind_data.npz",
    tau_percentile=75,
    alpha=3.0,
    # save npz (optional)
    save_path=None,
    # plot (optional)
    plot=False,
    plot_pdf_path="air_pollution_compact.pdf",
    plot_png_path="air_pollution_compact.png",
):
    # ============================================================
    # Random seed
    # ============================================================

    rng = np.random.default_rng(seed)

    # ============================================================
    # Grid (radians)
    # ============================================================

    lat = np.linspace(-np.pi/2, np.pi/2, n)
    lon = np.linspace(-np.pi, np.pi, n)

    Lon, Lat = np.meshgrid(lon, lat)

    # ============================================================
    # CONUS mask
    # ============================================================

    lon_min, lon_max = np.deg2rad(lon_bounds)
    lat_min, lat_max = np.deg2rad(lat_bounds)

    mask_us = (
        (Lon >= lon_min) &
        (Lon <= lon_max) &
        (Lat >= lat_min) &
        (Lat <= lat_max)
    )

    # ============================================================
    # Haversine distance
    # ============================================================

    def spherical_distance(lat1, lon1, lat2, lon2):

        dlat = lat1 - lat2
        dlon = lon1 - lon2

        a = (
            np.sin(dlat/2)**2 +
            np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        )

        a = np.clip(a, 0, 1)

        return 2*np.arcsin(np.sqrt(a))

    # ============================================================
    # Generate emission sources
    # ============================================================

    emission_lon = rng.uniform(lon_min, lon_max, H)
    emission_lat = rng.uniform(lat_min, lat_max, H)

    # ============================================================
    # Base air quality field
    # ============================================================

    aq = np.zeros_like(Lon)

    # decay scale = ~60 km

    sigma = (decay_km * 1000.0) / R_earth

    for h in range(H):

        dist = spherical_distance(
            Lat, Lon,
            emission_lat[h], emission_lon[h]
        )

        aq += np.exp(-(dist**2)/(2*sigma**2))

    # ============================================================
    # Load wind speed
    # ============================================================

    wind = np.load(wind_npz_path)

    lon_wind = wind["lon"]
    lat_wind = wind["lat"]
    speed_wind = wind["speed"]

    lon1d = np.deg2rad(lon_wind[0])
    lat1d = np.deg2rad(lat_wind[:,0])

    if lat1d[1] < lat1d[0]:

        lat1d = lat1d[::-1]
        speed_wind = speed_wind[::-1]

    interp_speed = RegularGridInterpolator(
        (lat1d, lon1d),
        speed_wind,
        bounds_error=False,
        fill_value=np.nan
    )

    points = np.stack([Lat.ravel(), Lon.ravel()], axis=1)

    speed = interp_speed(points).reshape(Lat.shape)

    aq_original = aq.copy()

    # ============================================================
    # Wind penalty
    # ============================================================

    speed_norm = (
        speed - np.nanmin(speed)
    ) / (
        np.nanmax(speed) - np.nanmin(speed)
    )

    speed_norm = np.nan_to_num(speed_norm)

    tau = np.nanpercentile(speed_norm[mask_us], tau_percentile)

    penalty = np.maximum(speed_norm - tau, 0)

    aq *= np.exp(-alpha * penalty)

    aq_wind = aq.copy()

    # ============================================================
    # Normalize
    # ============================================================

    aq -= np.nanmin(aq)

    aq /= np.nanmax(aq)

    aq[~mask_us] = np.nan

    # ============================================================
    # Save emission coords for plotting later
    # ============================================================

    emission_lat_deg = np.rad2deg(emission_lat)
    emission_lon_deg = np.rad2deg(emission_lon)

    # ============================================================
    # Save NPZ (optional)
    # ============================================================
    
    lat_deg = np.rad2deg(Lat).flatten()
    lon_deg = np.rad2deg(Lon).flatten()
    aq_flat = aq.flatten()

    mask = ~np.isnan(aq_flat)
    lat_deg = lat_deg[mask]
    lon_deg = lon_deg[mask]
    aq_flat = aq_flat[mask]


    if save_path is not None:
        np.savez(
            save_path,
            lon=lon_deg,
            lat=lat_deg,
            aq=aq_flat,
            aq_original=aq_original,
            aq_wind=aq_wind,
            speed=speed,
            mask_us=mask_us,
            emission_lon_deg=emission_lon_deg,
            emission_lat_deg=emission_lat_deg,
        )

    # ============================================================
    # Plot (optional)
    # ============================================================

    if plot:


        fig = go.Figure(go.Scattergeo())

        fig.add_trace(go.Scattergeo(
            lon=lon_deg, lat=lat_deg, mode="markers", showlegend=False,
            marker=dict(size=5, color=aq_flat, colorscale="Turbo", colorbar=dict(title="Air Quality"), opacity=0.6)
        ))


        fig.add_trace(go.Scattergeo(
            lon=emission_lon_deg, lat=emission_lat_deg, mode="markers", showlegend=False,
            marker=dict(size=2, color="Black", symbol="x"), name="Emission Sources"))

        fig.update_geos(
            visible=False, resolution=110, scope="usa",
            showcountries=True, countrycolor="Black",
            showsubunits=True, subunitcolor="Blue"
        )
        # fig.update_geos(projection_type="natural earth")

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=800, height=500)

        if plot_pdf_path is not None:
            fig.write_image(plot_pdf_path, scale=5)

        if plot_png_path is not None:
            fig.write_image(plot_png_path, scale=2)

        fig.show()

    out = {
        "Lat": Lat,
        "Lon": Lon,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "mask_us": mask_us,
        "aq": aq_flat,
        "aq_original": aq_original,
        "aq_wind": aq_wind,
        "speed": speed,
        "emission_lon": emission_lon,
        "emission_lat": emission_lat,
        "emission_lon_deg": emission_lon_deg,
        "emission_lat_deg": emission_lat_deg,
    }

    return out