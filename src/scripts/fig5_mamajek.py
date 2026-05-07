"""
Publication-ready script to generate disc fraction vs age comparison figure.

This script creates a figure comparing population synthesis predictions with
observational data from selected star-forming regions, showing disc fraction 
evolution over time for different photoevaporation prescriptions. The observational 
data combines low-UV (Michel et al. 2021, Pfalzner et al. 2022) regions.

Selected regions (from Manara et al. PPVII catalog): Lupus, Upper Scorpius, 
Chameleon I, ρ Ophiuchi, Chameleon II, Taurus, Corona Australis

Author: Giovanni Picogna
Date: 10.05.2023
Last Updated: March 2026 - Using Manara et al. star-forming regions
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import paths
from scipy.optimize import curve_fit
from lib import load_data
import os
import subprocess
import matplotlib.font_manager as font_manager

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================
# Configure matplotlib for publication quality
plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid rendering issues

# Try to use science plots style if available
try:
    import scienceplots
    plt.style.use('science')
    print("Using scienceplots style for publication quality")
except ImportError:
    print("scienceplots not available, using default matplotlib style")

sns.set_palette("pastel")  # Set seaborn color palette

kpse_cp = subprocess.run(['kpsewhich', '-var-value', 'TEXMFDIST'], capture_output=True, check=True)
font_loc1 = os.path.join(kpse_cp.stdout.decode('utf8').strip(), 'fonts', 'opentype', 'public', 'tex-gyre')
print(f'loading TeX Gyre fonts from "{font_loc1}"')
font_dirs = [font_loc1]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'TeX Gyre Termes'
plt.rcParams["mathtext.fontset"] = "stix"

# MNRAS style configuration
# Column width: 240pt = 10/3 inches for single column
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 8

plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid rendering issues

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Try to use Nimbus Roman font (MNRAS standard)
# try:
#     plt.rc('font', family='Nimbus Roman')
#    print("Using Nimbus Roman font (MNRAS standard)")
# except:
#    plt.rc('font', family='serif')
#    print("Nimbus Roman not available, using default serif font")

plt.rcParams["errorbar.capsize"] = 2

# Override specific parameters for MNRAS single-column figure
plt.rcParams.update({
    'figure.figsize': [10/3, 2.25],  # MNRAS single column width
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# ============================================================================
# FIGURE SETUP AND DATA LOADING
# ============================================================================
# Create figure optimized for MNRAS single-column (10/3 inches wide)
# Use GridSpec to create separate axes for legend and main plot
fig = plt.figure(figsize=(10/3, 2.25))
gs = gridspec.GridSpec(2, 1, height_ratios=[0.18, 1], hspace=0.15,
                       left=0.16, bottom=0.15, right=0.96, top=0.98)
ax_legend = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])

# Define data paths
data_path_internal = paths.data / "pop_spreading/"
data_path_internal_external = paths.data / "pop_spreading_extern/"
data_path_reduced_internal = paths.data / "pop_spreading_reduced/"
data_path_reduced_internal_external = paths.data / "pop_spreading_reduced_extern/"
data_path_norcrit = paths.data / "pop_spreading_reduced_norcrit/"

# Data filtering parameters
mask = True           # Apply masking to population synthesis data
mask_val = 1.e-13     # Threshold for disc mass masking

# ============================================================================
# POPULATION SYNTHESIS DATA PLOTTING
# ============================================================================
print("Loading population synthesis data...")

# Load and plot internal photoevaporation data
profile_internal = "Internal"
print(f"Loading internal photoevaporation data from: {data_path_internal}")
try:
    data_internal = load_data(data_path_internal, profile_name=profile_internal,
                         mask=mask, mask_val=mask_val)
    n_points_internal = len(data_internal['age'])
    print(f"Successfully loaded {n_points_internal} data points (internal)")
    
    ax.plot(data_internal["age"], data_internal["disc_fraction"], '.',
            color='#000000', markersize=2, label=profile_internal)
    
except Exception as e:
    print(f"Error loading internal photoevaporation data: {e}")
    # Create dummy data if loading fails
    dummy_ages = np.linspace(0.1, 15, 50)
    dummy_fractions = 100 * np.exp(-dummy_ages/3.0)
    ax.plot(dummy_ages, dummy_fractions, '.',
            color='#000000', markersize=2, label=profile_internal + " (dummy)")

# Load and plot extended discs data
profile_reduced_internal = "Reduced Internal"
print(f"Loading reduced internal photoevaporation data from: {data_path_reduced_internal}")
try:
    data_reduced_internal = load_data(data_path_reduced_internal, profile_name=profile_reduced_internal,
                         mask=mask, mask_val=mask_val)
    n_points_reduced_internal = len(data_reduced_internal['age'])
    print(f"Successfully loaded {n_points_reduced_internal} data points (reduced internal)")
    
    ax.plot(data_reduced_internal["age"], data_reduced_internal["disc_fraction"], '.',
            color='#56B4E9', markersize=2, label=profile_reduced_internal)
    
except Exception as e:
    print(f"Error loading reduced internal photoevaporation data: {e}")
    # Create dummy data if loading fails
    dummy_ages = np.linspace(0.1, 12, 50)
    dummy_fractions = 100 * np.exp(-dummy_ages/2.5)
    ax.plot(dummy_ages, dummy_fractions, '.',
            color='#56B4E9', markersize=2, label=profile_reduced_internal + " (dummy)")

# Load and plot extended discs data (reduced PE)
profile_reduced_internal_external = "Reduced Internal + External"
print(f"Loading reduced internal + external photoevaporation data from: {data_path_reduced_internal_external}")
try:
    data_reduced_internal_external = load_data(data_path_reduced_internal_external, profile_name=profile_reduced_internal_external,
                         mask=mask, mask_val=mask_val)
    n_points_reduced_internal_external = len(data_reduced_internal_external['age'])
    print(f"Successfully loaded {n_points_reduced_internal_external} data points (reduced internal + external)")
    
    ax.plot(data_reduced_internal_external["age"], data_reduced_internal_external["disc_fraction"], '.',
            color='#CC79A7', markersize=2, label=profile_reduced_internal_external)
    
except Exception as e:
    print(f"Error loading reduced internal + external photoevaporation data: {e}")
    # Create dummy data if loading fails
    dummy_ages = np.linspace(0.1, 12, 50)
    dummy_fractions = 100 * np.exp(-dummy_ages/2.5)
    ax.plot(dummy_ages, dummy_fractions, '.',
            color='#CC79A7', markersize=2, label=profile_reduced_internal_external + " (dummy)")

# Load and plot no critical radius data
profile_norcrit = f"Reduced internal without $r_1$"
print(f"Loading no critical radius photoevaporation data from: {data_path_norcrit}")
try:
    data_norcrit = load_data(data_path_norcrit, profile_name=profile_norcrit,
                         mask=mask, mask_val=mask_val)
    n_points_norcrit = len(data_norcrit['age'])
    print(f"Successfully loaded {n_points_norcrit} data points (no critical radius)")
    
    ax.plot(data_norcrit["age"], data_norcrit["disc_fraction"], '.',
            color='#E69F00', markersize=2, label=profile_norcrit)
    
except Exception as e:
    print(f"Error loading no critical radius photoevaporation data: {e}")
    # Create dummy data if loading fails
    dummy_ages = np.linspace(0.1, 12, 50)
    dummy_fractions = 100 * np.exp(-dummy_ages/2.5)
    ax.plot(dummy_ages, dummy_fractions, '.',
            color='#E69F00', markersize=2, label=profile_norcrit + " (dummy)")

# ============================================================================
# OBSERVATIONAL DATA OVERLAY
# ============================================================================
print("Loading observational data...")

# Load combined observational data from low-UV and high-UV datasets
# These provide comprehensive disc fraction measurements for selected regions
obs_file_low_uv = paths.data / "disc_fraction_low_UV.csv"
obs_file_high_uv = paths.data / "disc_fraction_high_UV.csv"

# Specific regions to include (from Manara et al. PPVII catalog)
selected_regions = [
    'Lupus', 'Upp Sco', 'Cham I', 'Ophiuchus', 'Cham II', 'Taurus', 'CrA'
]

print(f"Loading low-UV observational data from: {obs_file_low_uv}")
print(f"Loading high-UV observational data from: {obs_file_high_uv}")

# Read both datasets
data_low_uv = []
with open(obs_file_low_uv, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split(',')
        if len(parts) >= 7:
            name = parts[0]
            if name in selected_regions:
                data_low_uv.append(parts[:7])  # Include name and first 6 numerical columns

data_high_uv = []
with open(obs_file_high_uv, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split(',')
        if len(parts) >= 7:
            name = parts[0]
            if name in selected_regions:
                data_high_uv.append(parts[:7])  # Include name and first 6 numerical columns

# Combine datasets
all_obs_data = data_low_uv + data_high_uv

print(f"Selected {len(all_obs_data)} regions: {[d[0] for d in all_obs_data]}")

# Convert to arrays
sfr_names = np.array([d[0] for d in all_obs_data])
age_obs = np.array([float(d[1]) for d in all_obs_data])
age_lower_obs = np.array([float(d[2]) for d in all_obs_data])
age_upper_obs = np.array([float(d[3]) for d in all_obs_data])
frac_obs = np.array([float(d[4]) for d in all_obs_data])
frac_lower_obs = np.array([float(d[5]) for d in all_obs_data])
frac_upper_obs = np.array([float(d[6]) for d in all_obs_data])

# Calculate error bars
xerr_lower_obs = age_obs - age_lower_obs
xerr_upper_obs = age_upper_obs - age_obs
yerr_lower_obs = frac_obs - frac_lower_obs
yerr_upper_obs = frac_upper_obs - frac_obs

# Plot observational data
ax.errorbar(age_obs, frac_obs,
            xerr=[xerr_lower_obs, xerr_upper_obs], 
            yerr=[yerr_lower_obs, yerr_upper_obs],
            fmt='o', markersize=4, color='#D55E00', ecolor='darkred',
            markeredgewidth=0.5, markeredgecolor='darkred', elinewidth=1.5,
            capsize=2, capthick=1, alpha=0.8, zorder=3,
            label='Manara et al. regions')

mean_age_obs = np.mean(age_obs)

# ============================================================================
# E-FOLDING TIME CALCULATION FOR OBSERVED POPULATION
# ============================================================================
# Fit exponential decay model: f(t) = f0 * exp(-t/tau)
# where tau is the e-folding time
# Median disc lifetime is calculated as: t_median = tau * ln(2)
# (when 50% of the initial disc population has dispersed)

# Exponential decay functions
def exp_decay(t, f0, tau):
    """Exponential decay function: f(t) = f0 * exp(-t/tau)
    Parameters:
        f0: initial disc fraction (free parameter for observations)
        tau: e-folding time in Myr
    """
    return f0 * np.exp(-t / tau)

def predict_with_uncertainty(t, popt, pcov, n_samples=1000):
    """Generate prediction with uncertainty bounds from covariance matrix.
    Returns: mean prediction, lower bound (1-sigma), upper bound (1-sigma)
    """
    # Sample parameters from multivariate normal distribution
    param_samples = np.random.multivariate_normal(popt, pcov, n_samples)
    
    # Clip parameters to physical bounds to prevent unphysical behavior
    # f0 must be in [0, 100]% and tau must be positive
    param_samples[:, 0] = np.clip(param_samples[:, 0], 0, 100)  # f0 bounds
    param_samples[:, 1] = np.clip(param_samples[:, 1], 0.1, 50)  # tau > 0 (min 0.1 Myr)
    
    predictions = np.array([exp_decay(t, *params) for params in param_samples])
    
    # Calculate percentiles (68% confidence = 1-sigma)
    mean_pred = np.mean(predictions, axis=0)
    lower = np.percentile(predictions, 16, axis=0)
    upper = np.percentile(predictions, 84, axis=0)
    
    return mean_pred, lower, upper


# ----------------------------------------------------------------------------
# OBSERVATIONAL DATA EXPONENTIAL FIT
# ----------------------------------------------------------------------------
# Sort observational data by age for fitting
sort_indices_obs = np.argsort(age_obs)
age_sorted_obs = age_obs[sort_indices_obs]
frac_sorted_obs = frac_obs[sort_indices_obs]

# Fit exponential decay to observational data (FREE f0 and tau)
try:
    # Initial guess: f0 ~ 70%, tau ~ mean age
    initial_guess_obs = [70.0, mean_age_obs]
    
    # Perform curve fitting with both f0 and tau as free parameters
    popt_obs, pcov_obs = curve_fit(exp_decay, age_sorted_obs, frac_sorted_obs,
                                    p0=initial_guess_obs,
                                    bounds=([0, 0], [100, 50]),  # f0 in [0,100]%, tau in [0,50] Myr
                                    maxfev=5000)
    
    f0_fit_obs = popt_obs[0]
    tau_fit_obs = popt_obs[1]
    errors_obs = np.sqrt(np.diag(pcov_obs))
    f0_error_obs = errors_obs[0]
    tau_error_obs = errors_obs[1]
    
    # Calculate median disc lifetime (when 50% of discs have dispersed)
    t_median_obs = tau_fit_obs * np.log(2)
    t_median_error_obs = tau_error_obs * np.log(2)
    
    print(f"Observational fit: f0 = {f0_fit_obs:.1f} ± {f0_error_obs:.1f}%, τ = {tau_fit_obs:.2f} ± {tau_error_obs:.2f} Myr")
    print(f"Observational median disc lifetime: {t_median_obs:.2f} ± {t_median_error_obs:.2f} Myr")
    
    # Generate prediction with confidence bands
    age_fit = np.linspace(0.1, 20, 100)
    frac_fit_obs, frac_lower_obs_fit, frac_upper_obs_fit = predict_with_uncertainty(age_fit, popt_obs, pcov_obs)
    
    # Plot fitted curve with shaded confidence region
    ax.plot(age_fit, frac_fit_obs, '--', color='#D55E00', linewidth=2)
    ax.fill_between(age_fit, frac_lower_obs_fit, frac_upper_obs_fit, 
                    color='#D55E00', alpha=0.2, linewidth=0)
    
    # Add vertical line at median disc lifetime with uncertainty band
    ax.axvspan(t_median_obs - t_median_error_obs, t_median_obs + t_median_error_obs, 
               color='#D55E00', alpha=0.15, linewidth=0, zorder=0)
    ax.vlines(t_median_obs, 0, 90, color='#D55E00', ls=':', alpha=0.6, linewidth=1.5)
    
except Exception as e:
    print(f"Warning: Could not fit exponential decay to observational data: {e}")
    # Use simple estimate: tau ≈ mean age
    tau_estimate = mean_age_obs
    print(f"Using mean age as e-folding time estimate: {tau_estimate:.2f} Myr")

# ----------------------------------------------------------------------------
# MAMAJEK 2009 OBSERVATIONAL DATA FIT
# ----------------------------------------------------------------------------
print("\nLoading Mamajek et al. 2009 observational data...")
obs_file_mamajek = paths.data / "disc_fraction_Mamajek2009.csv"

# Read Mamajek 2009 dataset (all star-forming regions)
mamajek_data = []
with open(obs_file_mamajek, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split(',')
        if len(parts) >= 7:
            mamajek_data.append(parts[:7])  # Include name and first 6 numerical columns

print(f"Loaded {len(mamajek_data)} regions from Mamajek et al. 2009")

# Convert to arrays
mamajek_names = np.array([d[0] for d in mamajek_data])
age_mamajek = np.array([float(d[1]) for d in mamajek_data])
age_lower_mamajek = np.array([float(d[2]) for d in mamajek_data])
age_upper_mamajek = np.array([float(d[3]) for d in mamajek_data])
frac_mamajek = np.array([float(d[4]) for d in mamajek_data])
frac_lower_mamajek = np.array([float(d[5]) for d in mamajek_data])
frac_upper_mamajek = np.array([float(d[6]) for d in mamajek_data])

# Sort Mamajek data by age for fitting
sort_indices_mamajek = np.argsort(age_mamajek)
age_sorted_mamajek = age_mamajek[sort_indices_mamajek]
frac_sorted_mamajek = frac_mamajek[sort_indices_mamajek]

# Fit exponential decay to Mamajek data (FREE f0 and tau)
try:
    mean_age_mamajek = np.mean(age_mamajek)
    initial_guess_mamajek = [80.0, mean_age_mamajek]
    
    # Perform curve fitting
    popt_mamajek, pcov_mamajek = curve_fit(exp_decay, age_sorted_mamajek, frac_sorted_mamajek,
                                            p0=initial_guess_mamajek,
                                            bounds=([0, 0], [100, 50]),
                                            maxfev=5000)
    
    f0_fit_mamajek = popt_mamajek[0]
    tau_fit_mamajek = popt_mamajek[1]
    errors_mamajek = np.sqrt(np.diag(pcov_mamajek))
    f0_error_mamajek = errors_mamajek[0]
    tau_error_mamajek = errors_mamajek[1]
    
    # Calculate median disc lifetime
    t_median_mamajek = tau_fit_mamajek * np.log(2)
    t_median_error_mamajek = tau_error_mamajek * np.log(2)
    
    print(f"Mamajek et al. 2009 fit: f0 = {f0_fit_mamajek:.1f} ± {f0_error_mamajek:.1f}%, τ = {tau_fit_mamajek:.2f} ± {tau_error_mamajek:.2f} Myr")
    print(f"Mamajek et al. 2009 median disc lifetime: {t_median_mamajek:.2f} ± {t_median_error_mamajek:.2f} Myr")
    
    # Generate prediction with confidence bands
    age_fit_mamajek = np.linspace(0.1, 20, 100)
    frac_fit_mamajek, frac_lower_mamajek_fit, frac_upper_mamajek_fit = predict_with_uncertainty(
        age_fit_mamajek, popt_mamajek, pcov_mamajek)
    
    # Plot fitted curve with shaded confidence region (lighter color)
    ax.plot(age_fit_mamajek, frac_fit_mamajek, ':', color='#009E73', linewidth=2, 
            label='Mamajek et al. 2009 fit')
    ax.fill_between(age_fit_mamajek, frac_lower_mamajek_fit, frac_upper_mamajek_fit, 
                    color='#009E73', alpha=0.15, linewidth=0)
    
    # Add vertical line at median disc lifetime with uncertainty band
    ax.axvspan(t_median_mamajek - t_median_error_mamajek, t_median_mamajek + t_median_error_mamajek, 
               color='#009E73', alpha=0.1, linewidth=0, zorder=0)
    ax.vlines(t_median_mamajek, 0, 90, color='#009E73', ls=':', alpha=0.5, linewidth=1.5)
    
except Exception as e:
    print(f"Warning: Could not fit exponential decay to Mamajek data: {e}")

# ============================================================================
# E-FOLDING TIME CALCULATION FOR POPULATION SYNTHESIS DATA
# ============================================================================
# Calculate median disc lifetimes for all population synthesis models
# by fitting exponential decay and converting tau to t_median = tau * ln(2)
print("Calculating e-folding times for population synthesis data...")

# Calculate e-folding time for new prescription (critical radius dependence)
try:
    if 'data_internal' in locals():
        # Create binned data for fitting (reduce computational load)
        age_bins = np.linspace(0.1, 15, 50)
        bin_centers = (age_bins[1:] + age_bins[:-1]) / 2
        bin_fractions = []
        
        for i in range(len(age_bins)-1):
            mask_bin = ((data_internal["age"] >= age_bins[i]) & 
                       (data_internal["age"] < age_bins[i+1]))
            if np.sum(mask_bin) > 0:
                bin_fractions.append(np.mean(data_internal["disc_fraction"][mask_bin]))
            else:
                bin_fractions.append(0)
        
        bin_fractions = np.array(bin_fractions)
        valid_bins = bin_fractions > 0
        
        if np.sum(valid_bins) > 5:  # Need enough points for fitting
            # Fit exponential decay
            initial_guess = [100, 3.0]  # Initial guess for e-folding time
            popt_new, pcov_new = curve_fit(exp_decay, bin_centers[valid_bins], 
                                          bin_fractions[valid_bins],
                                          p0=initial_guess, maxfev=2000)
            
            f0_new, tau_new = popt_new
            tau_new_error = np.sqrt(np.diag(pcov_new))[1]
            
            # Calculate median disc lifetime
            t_median_new = tau_new * np.log(2)
            t_median_error_new = tau_new_error * np.log(2)
            
            print(f"Internal prescription e-folding time: {tau_new:.2f} ± {tau_new_error:.2f} Myr")
            print(f"Internal prescription median disc lifetime: {t_median_new:.2f} ± {t_median_error_new:.2f} Myr")
            
            # Add solid line for median disc lifetime
            ax.vlines(t_median_new, 0, 90, color='#000000', ls='-.', alpha=0.9, linewidth=2.0)
            
        else:
            print("New prescription: Not enough valid bins for fitting")

except Exception as e:
    print(f"Warning: Could not calculate e-folding time for new prescription: {e}")

# Calculate e-folding time for spreading discs prescription
try:
    if 'data_reduced_internal' in locals():
        # Create binned data for fitting (reduce computational load)
        age_bins = np.linspace(0.1, 10, 40)
        bin_centers = (age_bins[1:] + age_bins[:-1]) / 2
        bin_fractions = []
        
        for i in range(len(age_bins)-1):
            mask_bin = ((data_reduced_internal["age"] >= age_bins[i]) & 
                       (data_reduced_internal["age"] < age_bins[i+1]))
            if np.sum(mask_bin) > 0:
                bin_fractions.append(np.mean(data_reduced_internal["disc_fraction"][mask_bin]))
            else:
                bin_fractions.append(0)
        
        bin_fractions = np.array(bin_fractions)
        valid_bins = bin_fractions > 0
        
        if np.sum(valid_bins) > 5:  # Need enough points for fitting
            # Fit exponential decay
            initial_guess = [100, 2.0]  # Initial guess for e-folding time
            popt_spreading, pcov_spreading = curve_fit(exp_decay, bin_centers[valid_bins], 
                                          bin_fractions[valid_bins],
                                          p0=initial_guess, maxfev=2000)
            
            f0_spreading, tau_spreading = popt_spreading
            tau_spreading_error = np.sqrt(np.diag(pcov_spreading))[1]
            
            # Calculate median disc lifetime
            t_median_spreading = tau_spreading * np.log(2)
            t_median_error_spreading = tau_spreading_error * np.log(2)
            
            print(f"Reduced Internal prescription e-folding time: {tau_spreading:.2f} ± {tau_spreading_error:.2f} Myr")
            print(f"Reduced Internal prescription median disc lifetime: {t_median_spreading:.2f} ± {t_median_error_spreading:.2f} Myr")
            
            # Add dotted line for median disc lifetime
            ax.vlines(t_median_spreading, 0, 90, color='#56B4E9', ls='-.', alpha=0.9, linewidth=2.5)
            
        else:
            print("Spreading discs: Not enough valid bins for fitting")
            
except Exception as e:
    print(f"Warning: Could not calculate e-folding time for spreading discs: {e}")

# Calculate e-folding time for spreading discs (reduced PE) prescription
try:
    if 'data_reduced_internal_external' in locals():
        # Create binned data for fitting (reduce computational load)
        age_bins = np.linspace(0.1, 10, 40)
        bin_centers = (age_bins[1:] + age_bins[:-1]) / 2
        bin_fractions = []
        
        for i in range(len(age_bins)-1):
            mask_bin = ((data_reduced_internal_external["age"] >= age_bins[i]) & 
                       (data_reduced_internal_external["age"] < age_bins[i+1]))
            if np.sum(mask_bin) > 0:
                bin_fractions.append(np.mean(data_reduced_internal_external["disc_fraction"][mask_bin]))
            else:
                bin_fractions.append(0)
        
        bin_fractions = np.array(bin_fractions)
        valid_bins = bin_fractions > 0
        
        if np.sum(valid_bins) > 5:  # Need enough points for fitting
            # Fit exponential decay
            initial_guess = [100, 2.0]  # Initial guess for e-folding time
            popt_spreading_reduced, pcov_spreading_reduced = curve_fit(exp_decay, bin_centers[valid_bins], 
                                          bin_fractions[valid_bins],
                                          p0=initial_guess, maxfev=2000)
            
            f0_spreading_reduced, tau_spreading_reduced = popt_spreading_reduced
            tau_spreading_reduced_error = np.sqrt(np.diag(pcov_spreading_reduced))[1]
            
            # Calculate median disc lifetime
            t_median_spreading_reduced = tau_spreading_reduced * np.log(2)
            t_median_error_spreading_reduced = tau_spreading_reduced_error * np.log(2)
            
            print(f"Reduced Internal + External prescription e-folding time: {tau_spreading_reduced:.2f} ± {tau_spreading_reduced_error:.2f} Myr")
            print(f"Reduced Internal + External prescription median disc lifetime: {t_median_spreading_reduced:.2f} ± {t_median_error_spreading_reduced:.2f} Myr")
            
            # Add dash-dot line for median disc lifetime
            ax.vlines(t_median_spreading_reduced, 0, 90, color='#CC79A7', ls='-.', alpha=0.9, linewidth=2.0)
            
        else:
            print("Spreading discs (reduced PE): Not enough valid bins for fitting")
            
except Exception as e:
    print(f"Warning: Could not calculate e-folding time for spreading discs (reduced PE): {e}")

# Calculate e-folding time for no critical radius prescription
try:
    if 'data_norcrit' in locals():
        # Create binned data for fitting (reduce computational load)
        age_bins = np.linspace(0.1, 10, 40)
        bin_centers = (age_bins[1:] + age_bins[:-1]) / 2
        bin_fractions = []
        
        for i in range(len(age_bins)-1):
            mask_bin = ((data_norcrit["age"] >= age_bins[i]) & 
                       (data_norcrit["age"] < age_bins[i+1]))
            if np.sum(mask_bin) > 0:
                bin_fractions.append(np.mean(data_norcrit["disc_fraction"][mask_bin]))
            else:
                bin_fractions.append(0)
        
        bin_fractions = np.array(bin_fractions)
        valid_bins = bin_fractions > 0
        
        if np.sum(valid_bins) > 5:  # Need enough points for fitting
            # Fit exponential decay
            initial_guess = [100, 2.0]  # Initial guess for e-folding time
            popt_norcrit, pcov_norcrit = curve_fit(exp_decay, bin_centers[valid_bins], 
                                          bin_fractions[valid_bins],
                                          p0=initial_guess, maxfev=2000)
            
            f0_norcrit, tau_norcrit = popt_norcrit
            tau_norcrit_error = np.sqrt(np.diag(pcov_norcrit))[1]
            
            # Calculate median disc lifetime
            t_median_norcrit = tau_norcrit * np.log(2)
            t_median_error_norcrit = tau_norcrit_error * np.log(2)
            
            print(f"No critical radius prescription e-folding time: {tau_norcrit:.2f} ± {tau_norcrit_error:.2f} Myr")
            print(f"No critical radius prescription median disc lifetime: {t_median_norcrit:.2f} ± {t_median_error_norcrit:.2f} Myr")
            
            # Add dash-dot line for median disc lifetime
            ax.vlines(t_median_norcrit, 0, 90, color='#E69F00', ls='-.', alpha=0.9, linewidth=2.0)
            
        else:
            print("No critical radius: Not enough valid bins for fitting")
            
except Exception as e:
    print(f"Warning: Could not calculate e-folding time for no critical radius: {e}")

# ============================================================================
# PLOT FORMATTING AND FINALIZATION
# ============================================================================
print("Finalizing plot...")

# Configure legend and axis limits for publication quality
# Create legend in separate axis at top
handles, labels = ax.get_legend_handles_labels()
ax_legend.legend(handles, labels, loc='center', ncol=2, frameon=False,
                 columnspacing=0.6, handletextpad=0.3)
ax_legend.axis('off')  # Hide axis decorations
ax.set_xlim(0., 20.)
ax.set_ylim(0, 90.)

# Set axis labels with proper formatting
ax.set_ylabel(r'Disc fraction [$\%$]')
ax.set_xlabel('Age [Myr]')

# Improve tick formatting for publication
ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)
ax.tick_params(axis='both', which='minor', width=0.8)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Save figure with high resolution for publication
output_filename = paths.figures / 'Fig5.png'
fig.savefig(output_filename, format='png', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"Figure saved as {output_filename}")
fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
print(f"Figure dimensions: {fig_w:.1f}\" x {fig_h:.1f}\"")
print("Optimized for single-column A4 scientific publication")
print(f"Includes {len(sfr_names)} Manara et al. star-forming regions")
