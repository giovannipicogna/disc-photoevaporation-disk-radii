import numpy as np
import random
from scipy.constants import au
from scipy.stats.qmc import Sobol
import scipy.interpolate as interpolate
import paths
from scipy.stats import norm
import csv

random.seed(1234)  # for reproducability
m_earth_cgs = 5.972e27
m_sun_cgs = 1.9884e33
au_cgs = au*1e2

def save_to_csv(params, output_path=None):

    keys = params[0].keys()
    if output_path is None:
        output_path = paths.data / 'parameters.csv'
    with open(output_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(params)

def generate_random_value(measured_value, uncertainty):
  """
  Generates a random number from a normal distribution based on a measured value and its uncertainty.

  Args:
    measured_value (float): The central value of the measurement.
    uncertainty (float): The standard deviation (sigma) of the uncertainty.
                         This represents the spread of the possible values.

  Returns:
    float: A randomly generated number from the normal distribution.
  """
  # The numpy.random.normal() function generates a random number from a Gaussian distribution.
  # The 'loc' parameter is the mean (the measured_value).
  # The 'scale' parameter is the standard deviation (the uncertainty).
  # A normal distribution is a good model for random errors in measurement.
  return np.random.normal(loc=measured_value, scale=uncertainty)

# Regions within ~200 pc included in Anania et al. (2025)
_ANANIA_200PC_REGIONS = {'UppSco', 'Taurus', 'Lupus', 'rho_Oph', 'ChamI', 'ChamII', 'CrA'}


def sample_G0_anania2025(n_samples, seed=1234,
                         catalogue_path=None,
                         regions=None):
    """
    Sample G0 values from the empirical FUV-flux distribution of
    disc-hosting stars in nearby star-forming regions (d < 200 pc)
    measured by Anania et al. (2025), A&A 695, A74.

    The median FUV flux (FFUV, in Habing units G0) for each source is taken
    from the public CDS catalogue J/A+A/695/A74, filtered to the star-forming
    regions within ~200 pc: UppSco, Taurus, Lupus, rho_Oph, ChamI, ChamII, CrA.
    Sampling is done by drawing with replacement from the empirical distribution
    of log10(G0) using a KDE, so that intermediate values are covered smoothly.

    Args:
        n_samples (int): Number of G0 values to draw.
        seed (int): Random seed for reproducibility.
        catalogue_path (str or None): Path to the downloaded CDS TSV file.  If
            None, defaults to path_moon + 'data/anania2025_fuv.tsv'.
        regions (set or None): Set of region strings to include.  Defaults to
            _ANANIA_200PC_REGIONS (the 7 regions within ~200 pc).

    Returns:
        np.ndarray: Array of G0 values (Habing units).
    """
    import pandas as pd
    from scipy.stats import gaussian_kde

    if catalogue_path is None:
        catalogue_path = paths.data / 'anania2025_fuv.tsv'
    if regions is None:
        regions = _ANANIA_200PC_REGIONS

    # ── Read CDS TSV (skip comment lines starting with '#') ────────────────
    df = pd.read_csv(catalogue_path, sep='\t', comment='#',
                     skipinitialspace=True)
    # Strip whitespace from column names and Region values
    df.columns = df.columns.str.strip()
    df['Region'] = df['Region'].str.strip()

    # Keep only the 200 pc regions and rows with a valid FFUV value
    df = df[df['Region'].isin(regions)].copy()
    df['FFUV'] = pd.to_numeric(df['FFUV'], errors='coerce')
    df = df.dropna(subset=['FFUV'])
    df = df[df['FFUV'] > 0]

    print(f"Anania+2025 catalogue: {len(df)} sources in 200 pc regions")
    print(f"  Regions included: {sorted(df['Region'].unique())}")
    print(f"  G0 range: {df['FFUV'].min():.2f} – {df['FFUV'].max():.2f}")
    print(f"  Median G0: {df['FFUV'].median():.2f}")
    print(f"  Log10(median): {np.log10(df['FFUV'].median()):.2f}")

    # ── KDE in log space, then sample ─────────────────────────────────────
    np.random.seed(seed)
    log_g0 = np.log10(df['FFUV'].values)
    kde = gaussian_kde(log_g0)

    # Draw samples in log space and convert back
    log_samples = kde.resample(n_samples, seed=seed)[0]
    # Clip to a physically reasonable range [1, 1e6]
    log_samples = np.clip(log_samples, 0.0, 6.0)
    return 10.0**log_samples


def generate_initial_parameters(n_runs):

    params = []
    params_out_of_range = []

    Lx_func_025 = paths.data / 'LxfuncONC025.dat'
    Lx_func_05 = paths.data / 'LxfuncONC05.dat'
    Lx_func_1 = paths.data / 'LxfuncONC1.dat'
    # cumulative density function Guedel+2007 (inverted Kaplan-Maier-estimator)
    lxs025, cdf025 = np.loadtxt(Lx_func_025, unpack=True, comments='#')
    lxs05, cdf05 = np.loadtxt(Lx_func_05, unpack=True, comments='#')
    lxs1, cdf1 = np.loadtxt(Lx_func_1, unpack=True, comments='#')

    n_sobol = n_runs * 4   # oversample to survive the r1 filter
    r1_min_au = 1.0        # minimum critical radius [AU]
    r1_max_au = 500.0      # maximum critical radius [AU]

    # get IMF from Kroupa+2001
    IMF_Kroupa = []
    masses = np.linspace(0.1, 1.0, n_runs)

    for i in range(len(masses)):
        if masses[i] < 0.5:
            IMF_Kroupa.append([masses[i], masses[i]**(-1.3)])
        elif masses[i] >= 0.5:
            IMF_Kroupa.append([masses[i], masses[i]**(-2.3)])

    IMF_Kroupa = np.array(IMF_Kroupa)
    weights = IMF_Kroupa[:, 1]

    # Sample G0 from the empirical Anania+2025 distribution for the
    # 200 pc star-forming regions (UppSco, Taurus, Lupus, rho_Oph,
    # ChamI, ChamII, CrA).  This is more realistic than the Schechter
    # parametrisation of Winter & Haworth (2022) for nearby populations.
    G0_samples = sample_G0_anania2025(n_runs, seed=1234)

    # Build G0 KDE inverse CDF from Anania+2025 data (for Sobol quantile inversion)
    import pandas as pd
    from scipy.stats import gaussian_kde
    _cat_path = paths.data / 'anania2025_fuv.tsv'
    _df = pd.read_csv(_cat_path, sep='\t', comment='#', skipinitialspace=True)
    _df.columns = _df.columns.str.strip()
    _df['Region'] = _df['Region'].str.strip()
    _df = _df[_df['Region'].isin(_ANANIA_200PC_REGIONS)].copy()
    _df['FFUV'] = pd.to_numeric(_df['FFUV'], errors='coerce')
    _df = _df.dropna(subset=['FFUV'])
    _df = _df[_df['FFUV'] > 0]
    _log_g0_data = np.log10(_df['FFUV'].values)
    _kde_g0 = gaussian_kde(_log_g0_data)
    _log_g0_grid = np.linspace(-0.5, 6.5, 20000)
    _kde_g0_cdf = np.cumsum(_kde_g0(_log_g0_grid)) * (_log_g0_grid[1] - _log_g0_grid[0])
    _kde_g0_cdf /= _kde_g0_cdf[-1]
    inv_cdf_g0 = interpolate.interp1d(_kde_g0_cdf, _log_g0_grid, kind='linear',
                                       bounds_error=False,
                                       fill_value=(_log_g0_grid[0], _log_g0_grid[-1]))

    # Define parameter ranges for Sobol sampling (9 dimensions)
    # Dimensions 0-6: disc parameters (scaled to physical ranges below)
    # Dimension 7: L_x quantile — uniform [0,1] passed to ONC CDF inversion
    # Dimension 8: G0 quantile  — uniform [0,1] passed to Anania+2025 KDE inversion
    param_ranges = {
        'alpha_mdust': (1.2 - 3*0.2, 1.2 + 3*0.2),  # (0.6, 1.8)
        'beta_mdust': (1.8 - 3*0.4, 1.8 + 3*0.4),   # (0.6, 3.0)
        'delta_mdust': (0.9 - 3*0.1, 0.9 + 3*0.1),  # (0.6, 1.2)
        'alpha_m': (1.7 - 3*0.2, 1.7 + 3*0.2),      # (1.1, 2.3)
        'beta_m_log': (-4.2 - 3*0.2, -4.2 + 3*0.2), # (-4.8, -3.6)
        'alpha_log': (-4.0, -2.0),
        'mdust_scatter_uniform': (0.0, 1.0),          # Will be transformed to normal
        'lx_quantile': (0.0, 1.0),                    # Uniform quantile → L_x via ONC CDF
        'g0_quantile': (0.0, 1.0),                    # Uniform quantile → G0 via Anania+2025 KDE
    }

    # Initialize Sobol sampler for 9 dimensions
    n_dims = 9
    sobol = Sobol(d=n_dims, seed=1234)

    # Generate oversampled Sobol pool to survive the r1 filter
    sobol_samples = sobol.random(n_sobol)

    # Scale Sobol samples to parameter ranges
    param_samples = np.zeros((n_sobol, n_dims))
    for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
        param_samples[:, i] = min_val + (max_val - min_val) * sobol_samples[:, i]

    # Extract parameter arrays
    alpha_mdust_samples = param_samples[:, 0]
    beta_mdust_samples = param_samples[:, 1]
    delta_mdust_samples = param_samples[:, 2]
    alpha_m_samples = param_samples[:, 3]
    beta_m_log_samples = param_samples[:, 4]
    alpha_log_samples = param_samples[:, 5]
    mdust_scatter_uniform = param_samples[:, 6]
    lx_quantile_samples  = param_samples[:, 7]   # Sobol quantile → L_x
    g0_quantile_samples  = param_samples[:, 8]   # Sobol quantile → G0

    disc_to_star_mass_slope = [generate_random_value(1.8, 0.4) for _ in range(n_sobol)]
    disc_to_star_mass_intercept = [generate_random_value(1.2, 0.2) for _ in range(n_sobol)]
    disc_to_star_mass_delta = [generate_random_value(0.9, 0.1) for _ in range(n_sobol)]
    mass_radius_slope = [generate_random_value(1.7, 0.2) for _ in range(n_sobol)]
    mass_radius_intercept = [generate_random_value(-4.2, 0.2) for _ in range(n_sobol)]

    for i in range(n_sobol):
        if len(params) >= n_runs:
            break

        params_run = {}

        #############################
        # Star properties
        #############################

        Mstar = np.random.choice(masses, size=1, p=weights/np.sum(weights),
                                 replace=True)[0]
        params_run["mstar"] = Mstar

        # Star X-ray luminosity
        # Use Sobol quantile (dim 7) for CDF inversion — better coverage than np.random
        rand_val = float(lx_quantile_samples[i])

        if Mstar <= 0.25:
            lxs_rescaled = lxs025 + np.log10((Mstar/0.16)**1.54)
            # Clamp random value to CDF bounds
            rand_val_clamped = np.clip(rand_val, cdf025.min(), cdf025.max())
            inv_cdf025 = interpolate.interp1d(cdf025, lxs_rescaled,
                                              kind='linear',
                                              bounds_error=False,
                                              fill_value=(lxs_rescaled[0], lxs_rescaled[-1]))
            lx = inv_cdf025(rand_val_clamped)
        elif Mstar <= 0.5:
            lxs_rescaled = lxs05 + np.log10((Mstar/0.36)**1.54)
            # Clamp random value to CDF bounds
            rand_val_clamped = np.clip(rand_val, cdf05.min(), cdf05.max())
            inv_cdf05 = interpolate.interp1d(cdf05, lxs_rescaled,
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=(lxs_rescaled[0], lxs_rescaled[-1]))
            lx = inv_cdf05(rand_val_clamped)
        else:
            lxs_rescaled = lxs1 + np.log10((Mstar/0.7)**1.54)
            # Clamp random value to CDF bounds
            rand_val_clamped = np.clip(rand_val, cdf1.min(), cdf1.max())
            inv_cdf1 = interpolate.interp1d(cdf1, lxs_rescaled,
                                            kind='linear',
                                            bounds_error=False,
                                            fill_value=(lxs_rescaled[0], lxs_rescaled[-1]))
            lx = inv_cdf1(rand_val_clamped)

        params_run["L_x"] = 10.**(lx - 30.)

        # Validate L_x to prevent NaN propagation in photoevaporation
        import math
        if math.isnan(params_run["L_x"]) or not math.isfinite(params_run["L_x"]):
            print(f"❌ ERROR: Invalid L_x value generated: {params_run['L_x']}")
            print(f"   Source lx value: {lx}")
            print(f"   Mstar: {Mstar}")
            print("   Skipping this simulation due to invalid X-ray luminosity")
            continue  # Skip this iteration

        #############################
        # FUV field strength (G0)
        #############################
        
        # G0 from Anania+2025 KDE via Sobol quantile (dim 8) — better coverage than resample
        _log_g0_i = float(inv_cdf_g0(float(g0_quantile_samples[i])))
        params_run["Habing_G0"] = float(np.clip(10.0**_log_g0_i, 1.0, 1e6))
        
        #################################################
        # Disc properties (using Sobol sampled parameters)
        #################################################

        # Dust disk mass - using Sobol-sampled parameters
        alpha_mdust = alpha_mdust_samples[i]
        beta_mdust = beta_mdust_samples[i]
        delta_mdust = delta_mdust_samples[i]

        log_Mdust_disc = alpha_mdust + beta_mdust * np.log10(params_run["mstar"])

        # Transform uniform [0,1] to normal distribution N(0, delta_mdust)
        # Using inverse CDF (probit) transformation
        mdust_scatter = norm.ppf(mdust_scatter_uniform[i]) * delta_mdust

        Mdust = 10**(log_Mdust_disc + mdust_scatter)

        # Gas disk mass
        params_run["m0"] = 100.*Mdust*m_earth_cgs/m_sun_cgs

        # Disk critical radius - using Sobol-sampled parameters
        alpha_m = alpha_m_samples[i]
        beta_m = 10**(beta_m_log_samples[i])
        params_run["r1"] = (params_run["m0"]/beta_m)**(1./alpha_m)

        # Filter on critical radius: keep discs outside range separately
        if params_run["r1"] < r1_min_au or params_run["r1"] > r1_max_au:
            params_out_of_range.append(params_run.copy())
            continue

        # Alpha parameter - using Sobol-sampled value
        alpha = alpha_log_samples[i]
        params_run["alpha"] = 10**alpha

        params.append(params_run.copy())

    return params, params_out_of_range

if __name__ == '__main__':
    n_runs = 4096
    print(f"Generating {n_runs} parameter sets using Sobol sequence...")
    params, params_out_of_range = generate_initial_parameters(n_runs)
    print(f"Successfully generated {len(params)} parameter sets")
    save_to_csv(params)
    print(f"Parameters saved to {paths.data / 'parameters.csv'}")
    if params_out_of_range:
        oor_path = paths.data / 'parameters_out_of_range.csv'
        save_to_csv(params_out_of_range, oor_path)
        print(f"Out-of-range parameters ({len(params_out_of_range)}) saved to {oor_path}")
