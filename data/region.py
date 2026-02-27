import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors


def generate_regional_data(
    level="postal",
    N=None,
    seed=111,
    lon_bounds=(-125, -66),
    lat_bounds=(25, 49),
    k=5,
    states_geojson_url="https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
):

    # --------------------------------------------------
    # Choose sampling size
    # --------------------------------------------------

    if N is None:
        if level == "postal":
            N = 50000
        elif level == "county":
            N = 5500
        else:
            raise ValueError('level must be "postal" or "county"')

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Generate candidate points
    # --------------------------------------------------

    lon = rng.uniform(lon_bounds[0], lon_bounds[1], N)
    lat = rng.uniform(lat_bounds[0], lat_bounds[1], N)

    # --------------------------------------------------
    # Load US states
    # --------------------------------------------------

    states = gpd.read_file(states_geojson_url)

    # --------------------------------------------------
    # Create GeoDataFrame of points
    # --------------------------------------------------

    points = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(lon, lat)],
        crs="EPSG:4326"
    )

    # --------------------------------------------------
    # Spatial join
    # --------------------------------------------------

    points_with_states = gpd.sjoin(
        points,
        states,
        how="inner",
        predicate="within"
    ).reset_index(drop=True)

    print("Points inside US:", len(points_with_states))

    # --------------------------------------------------
    # Extract coords and state labels
    # --------------------------------------------------

    coords = np.column_stack([
        points_with_states.geometry.x.values,
        points_with_states.geometry.y.values
    ])

    state_labels = points_with_states["name"].astype("category").cat.codes

    # --------------------------------------------------
    # Build kNN graph
    # --------------------------------------------------

    coords_rad = np.radians(coords[:, [1, 0]])

    nbrs = NearestNeighbors(
        n_neighbors=k+1,
        metric="haversine"
    ).fit(coords_rad)

    distances, indices = nbrs.kneighbors(coords_rad)

    edges = set()

    for i in range(len(coords)):
        for j in indices[i][1:]:
            edges.add(tuple(sorted((i, j))))

    edges = list(edges)

    print("Edges:", len(edges))

    return {
        "coords": coords,
        "state_labels": state_labels,
        "edges": edges,
        "points_gdf": points_with_states,
        "states_gdf": states,
        "level": level,
        "N_requested": N,
        "N_kept": len(coords),
    }