import numpy as np
import random
from scipy.constants import au
from scipy.stats.qmc import Sobol
import scipy.interpolate as interpolate
import paths
import csv

random.seed(1234)  # for reproducability
m_earth_cgs = 5.972e27
m_sun_cgs = 1.9884e33
au_cgs = au*1e2

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


def save_to_csv(params):

    keys = params[0].keys()
    output_file = paths.data / 'parameters.csv'
    with open(output_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(params)


def generate_initial_parameters(n_runs):

    params = []

    Lx_func_025 = paths.data / 'LxfuncONC025.dat'
    Lx_func_05 = paths.data / 'LxfuncONC05.dat'
    Lx_func_1 = paths.data / 'LxfuncONC1.dat'
    # cumulative density function Guedel+2007 (inverted Kaplan-Maier-estimator)
    lxs025, cdf025 = np.loadtxt(Lx_func_025, unpack=True, comments='#')
    lxs05, cdf05 = np.loadtxt(Lx_func_05, unpack=True, comments='#')
    lxs1, cdf1 = np.loadtxt(Lx_func_1, unpack=True, comments='#')

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

    # Define parameter ranges for Sobol sampling
    # We sample 7 parameters with Sobol sequence:
    # 1. disc_to_star_mass_intercept (alpha_mdust): mean=1.2, sigma=0.2 -> sample in range [mean-3*sigma, mean+3*sigma]
    # 2. disc_to_star_mass_slope (beta_mdust): mean=1.8, sigma=0.4 -> sample in range [mean-3*sigma, mean+3*sigma]
    # 3. disc_to_star_mass_delta (delta_mdust): mean=0.9, sigma=0.1 -> sample in range [mean-3*sigma, mean+3*sigma]
    # 4. mass_radius_slope (alpha_m): mean=1.7, sigma=0.2 -> sample in range [mean-3*sigma, mean+3*sigma]
    # 5. mass_radius_intercept (beta_m as log10): mean=-4.2, sigma=0.2 -> sample in range [mean-3*sigma, mean+3*sigma]
    # 6. alpha (as log10): range [-4, -2]
    # 7. Mdust scatter: N(0, delta_mdust) -> uniform [0, 1] to transform to normal later
    
    param_ranges = {
        'alpha_mdust': (1.2 - 3*0.2, 1.2 + 3*0.2),  # (0.6, 1.8)
        'beta_mdust': (1.8 - 3*0.4, 1.8 + 3*0.4),   # (0.6, 3.0)
        'delta_mdust': (0.9 - 3*0.1, 0.9 + 3*0.1),  # (0.6, 1.2)
        'alpha_m': (1.7 - 3*0.2, 1.7 + 3*0.2),      # (1.1, 2.3)
        'beta_m_log': (-4.2 - 3*0.2, -4.2 + 3*0.2), # (-4.8, -3.6)
        'alpha_log': (-4.0, -2.0),
        'mdust_scatter_uniform': (0.0, 1.0)  # Will be transformed to normal
    }
    
    # Initialize Sobol sampler for 7 dimensions
    n_dims = 7
    sobol = Sobol(d=n_dims, seed=1234)
    
    # Generate Sobol samples in [0, 1]^n_dims
    sobol_samples = sobol.random(n_runs)
    
    # Scale Sobol samples to parameter ranges
    param_samples = np.zeros((n_runs, n_dims))
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
    
    for i in range(n_runs):

        params_run = {}

        Mstar = np.random.choice(masses, size=1, p=weights/np.sum(weights),
                                 replace=True)[0]
        params_run["mstar"] = Mstar

        # Star X-ray luminosity
        # Generate random value and clamp it to valid CDF range to avoid extrapolation
        rand_val = np.random.random()
        
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

        #################################################
        # Disc properties (using Sobol samples)
        #################################################

        # Dust disk mass - using Sobol-sampled parameters
        alpha_mdust = alpha_mdust_samples[i]
        beta_mdust = beta_mdust_samples[i]
        delta_mdust = delta_mdust_samples[i]
        
        log_Mdust_disc = alpha_mdust + beta_mdust * np.log10(params_run["mstar"])
        
        # Transform uniform [0,1] to normal distribution N(0, delta_mdust)
        # Using inverse CDF (probit) transformation
        from scipy.stats import norm
        mdust_scatter = norm.ppf(mdust_scatter_uniform[i]) * delta_mdust
        
        Mdust = 10**(log_Mdust_disc + mdust_scatter)

        # Gas disk mass
        params_run["m0"] = 100.*Mdust*m_earth_cgs/m_sun_cgs

        # Disk critical radius - using Sobol-sampled parameters
        alpha_m = alpha_m_samples[i]
        beta_m = 10**(beta_m_log_samples[i])
        params_run["r1"] = (params_run["m0"]/beta_m)**(1./alpha_m)

        # Alpha parameter - using Sobol-sampled value
        alpha = alpha_log_samples[i]
        params_run["alpha"] = 10**alpha

        params.append(params_run.copy())        

    return params 


if __name__ == '__main__':
    n_runs = 10000
    print(f"Generating {n_runs} parameter sets using Sobol sequence...")
    params = generate_initial_parameters(n_runs)
    print(f"Successfully generated {len(params)} parameter sets")
    save_to_csv(params)
    print(f"Parameters saved to {paths.data / 'parameters.csv'}")
