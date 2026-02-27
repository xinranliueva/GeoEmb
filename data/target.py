# ============================================================
# Generate two DIFFERENT regression targets with interpretation
# ============================================================

import numpy as np


def generate_regression_targets(coords, Xw, Xa, seed=0):
    
    """
    Generates two interpretable region-level regression targets:

    Target 1: Respiratory Health Risk Index
        influenced by pollution exposure and wind-driven transport

    Target 2: Environmental Exposure Burden
        influenced by accumulated pollution and spatial environmental factors

    coords : (N,2) coordinates (lon, lat)
    Xw     : wind modality features
    Xa     : air quality modality features

    returns:
        targets : (N,2)
    """
    rng = np.random.default_rng(seed)

    N = coords.shape[0]


    # ========================================================
    # Smooth spatial field (shared latent geography)
    #
    # Interpretation:
    # underlying geographic vulnerability / ecosystem baseline
    # ========================================================

    C = coords - coords.mean(axis=0)
    W = rng.normal(size=(2, 32)) / 3.0
    b = rng.uniform(0, 2*np.pi, size=32)
    spatial = np.cos(C @ W + b) @ rng.normal(size=32)
    spatial = spatial - spatial.min()
    spatial = spatial / (spatial.max() + 1e-8)

    # ========================================================
    # Wind effect (Speed only)
    # ========================================================

    wind = Xw[:, -1]
    wind = wind / (wind.max() + 1e-8)


    # ========================================================
    # AQ effect
    # ========================================================

    aq = (Xa / (Xa.max() + 1e-8)).squeeze()

    # ========================================================
    # TARGET 1: Environmental Burden
    #
    # Interpretation:
    #
    # Environmental burden represents cumulative ecosystem exposure
    # driven directly by pollution and wind transport.
    # - AQ is the primary driver representing pollutant concentration
    # - Wind transports and redistributes pollutants across regions
    # - AQ×wind captures transported pollution amplifying accumulation
    # - Spatial term represents baseline geographic accumulation tendency
    # - Noise represents unobserved environmental variability
    # ========================================================

    noise1 = rng.gamma(shape=2.0, scale=0.05, size=N)
    environmental_burden = 2.5*aq + 2.5*wind + 1.0*aq*wind + 0.8*spatial + 0.2*noise1


    # ========================================================
    # TARGET 2: Respiratory Health Risk
    #
    # Interpretation:
    #
    # Respiratory health risk reflects physiological response to pollutant exposure.
    # - AQ contributes nonlinearly reflecting dose–response effects
    # - Environmental burden contributes as cumulative exposure history
    # - Wind contributes by influencing acute exposure conditions
    # - Spatial term represents baseline geographic and population vulnerability
    # - Threshold effect represents increased risk above elevated pollution levels
    # - Noise represents individual-level and demographic variability
    # ========================================================

    noise2 = rng.gamma(shape=2.0, scale=0.05, size=N)

    respiratory_risk = (
        0.5 * environmental_burden
        + 1.2 * aq**2
        + 1.0 * np.sqrt(np.abs(wind))
        + 0.1 * spatial
        + 1.0 * np.maximum(aq - np.percentile(aq, 70), 0)
        + 0.1*noise2
    )


    # ========================================================
    # Stack and standardize
    # ========================================================

    targets = np.stack([environmental_burden, respiratory_risk], axis=1)


    return targets