"""
Publication-ready script to generate Figure 7: Critical radius distribution comparison.

This script creates a comprehensive figure comparing photoevaporation prescriptions
through the critical radius (r1) distribution as a function of time:
- 2D histograms of population synthesis results (age vs critical radius)
- KDE marginal distributions showing prescription differences

While external photoevaporation has little effect on disc lifetime and accretion rates
(due to low mean G0 values), it should have a stronger influence on the disc outer radius.

The figure layout consists of:
- Panel a: Internal Photoevaporation only
- Panel b: Internal + External Photoevaporation
- Bottom: Age-marginalized KDE differences
- Right: Critical radius-marginalized KDE differences

Author: Giovanni Picogna
Date: March 2026
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import paths
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
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

sns.set_palette("pastel")            # Set seaborn color palette

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
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 8

plt.rcParams['text.usetex'] = False

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

plt.rcParams["errorbar.capsize"] = 2

# Override specific parameters for MNRAS single-column figure
plt.rcParams.update({
    'figure.figsize': [20./3., 3.0],  # MNRAS single column width, reduced height
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================
def load_data_with_r1(path, profile_name="Full sample", mask=True, mask_val=1.e-11, binary_fraction=0.876):
    """
    Load population synthesis data including critical radius (r1).
    
    r1.dat now contains time series data for all simulations (like Macc.dat, age.dat, frac.dat).
    
    Parameters:
    -----------
    path : pathlib.Path
        Path to the data directory
    profile_name : str
        Name identifier for this dataset
    mask : bool
        Whether to apply masking to accretion rate data
    mask_val : float
        Threshold value for accretion rate masking
    binary_fraction : float
        Binary fraction correction factor
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: mdot_acc, disk_fraction, age, r1, profile
    """
    # Load time series data
    Macc_arr = np.loadtxt(str(path)+"/Macc.dat")
    age_arr = np.loadtxt(str(path)+"/age.dat")
    disk_frac = np.loadtxt(str(path)+"/frac.dat")
    r1_arr = np.loadtxt(str(path)+"/r1.dat")  # Time series data like Macc.dat
    
    print(f"  Loading {profile_name}:")
    print(f"    Total data points: {len(age_arr)}")
    
    # Ensure all arrays have the same length
    min_length = min(len(age_arr), len(Macc_arr), len(disk_frac), len(r1_arr))
    age = age_arr[:min_length] / 1e6  # Convert to Myr
    frac = disk_frac[:min_length] * binary_fraction
    mdot_acc = Macc_arr[:min_length]
    r1 = r1_arr[:min_length]
    
    profile = np.array([profile_name] * min_length)
    
    data = pd.DataFrame(
        {
            "mdot_acc": mdot_acc,
            "disk_fraction": frac,
            "age": age,
            "r1": r1,
            "profile": profile
        }
    )
    
    if mask is True:
        mask_accretion = (data["mdot_acc"] > mask_val)
        data = data[mask_accretion]
    
    return data


# ============================================================================
# FIGURE SETUP AND LAYOUT
# ============================================================================
# Create main figure optimized for MNRAS two-column layout
fig = plt.figure(figsize=(20./3., 3.0))

# Create sophisticated gridspec layout:
# - Column 0: Dedicated colorbar (width_ratio=0.15)
# - Columns 1-2: Main plots (width_ratio=2 each)
# - Column 3: KDE r1 plot (width_ratio=1.0)
gs = gridspec.GridSpec(1, 4,
                       width_ratios=[0.15, 2, 2, 1.],
                       left=0.06, bottom=0.12, right=0.98, top=0.95,
                       wspace=0.05)

# Define subplot axes
cbar_ax_models = fig.add_subplot(gs[0, 0])  # Dedicated colorbar axis

# Main comparison panels
ax_main = [
    fig.add_subplot(gs[0, 1]),  # Panel a: Internal only
    fig.add_subplot(gs[0, 2])   # Panel b: Internal + External
]

# KDE r1 plot shares y-axis with main panels
ax_kde_r1 = fig.add_subplot(gs[0, 3], sharey=ax_main[0])

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
# Data filtering parameters
mask = False          # No additional masking applied
mask_val = 1.e-15     # Threshold value for potential masking

# Load population synthesis data for internal photoevaporation only
path = paths.data / 'pop_spreading_reduced/'
profile_name = "Internal Photoevaporation"
data_internal = load_data_with_r1(path, profile_name=profile_name, mask=mask,
                                   mask_val=mask_val)

# Load population synthesis data for internal + external photoevaporation
path = paths.data / 'pop_spreading_reduced_extern/'
profile_name = "Internal + External Photoevaporation"
data_external = load_data_with_r1(path, profile_name=profile_name, mask=mask,
                                   mask_val=mask_val)

# Print diagnostic information about loaded data
print(f"\nData loaded:")
print(f"Internal photoevaporation data points: {len(data_internal)}")
print(f"Internal + External photoevaporation data points: {len(data_external)}")
if len(data_internal) > 0:
    print(f"Internal age range: {data_internal['age'].min():.2f} - {data_internal['age'].max():.2f} Myr")
    print(f"Internal r1 range: {data_internal['r1'].min():.2f} - {data_internal['r1'].max():.2f} AU")
else:
    print("WARNING: Internal dataset is empty!")
if len(data_external) > 0:
    print(f"External age range: {data_external['age'].min():.2f} - {data_external['age'].max():.2f} Myr")
    print(f"External r1 range: {data_external['r1'].min():.2f} - {data_external['r1'].max():.2f} AU")
else:
    print("WARNING: External dataset is empty!")

# ============================================================================
# PLOTTING EXECUTION
# ============================================================================
# Determine appropriate r1 limits for plotting
# Use full radial range to capture all physics
r1_min = 1e0  # AU - minimum for log scale
r1_max = 3.5e1   # AU - maximum outer disc radius
print(f"\nr1 range for plots: {r1_min:.3e} - {r1_max:.1e} AU")

# Filter data to focus on typical disc radii
data_internal_plot = data_internal[(data_internal['r1'] >= r1_min) & (data_internal['r1'] <= r1_max)]
data_external_plot = data_external[(data_external['r1'] >= r1_min) & (data_external['r1'] <= r1_max)]

print(f"After filtering to {r1_min:.3e} - {r1_max:.1e} AU:")
print(f"  Internal: {len(data_internal_plot)} data points ({100*len(data_internal_plot)/len(data_internal):.1f}%)")
print(f"  External: {len(data_external_plot)} data points ({100*len(data_external_plot)/len(data_external):.1f}%)")

# Plot first panel: internal photoevaporation only
sns.histplot(data=data_internal_plot, x="age", y="r1", bins=(100, 100),
             cbar=True, stat='density',
             cmap='turbo', norm=LogNorm(vmin=1.e-3, vmax=1.e-1),
             vmin=None, vmax=None, log_scale=(False, True),
             cbar_kws={'orientation': 'vertical', 'label': 'density'},
             cbar_ax=cbar_ax_models,
             kde=True, ax=ax_main[0])

# Configure the colorbar properly with title and ticks on the left
cbar_ax_models.set_ylabel('Model density', rotation=90, fontsize=10)
cbar_ax_models.yaxis.set_label_position('left')
cbar_ax_models.yaxis.set_ticks_position('left')
cbar_ax_models.tick_params(labelsize=6, which='both', left=True, right=False)

# Configure panel a
ax_main[0].set_ylabel("")
ax_main[0].tick_params(left=False, labelleft=False)
ax_main[0].set_xlim(0., 20.)
ax_main[0].set_ylim(r1_min, r1_max)
ax_main[0].set_yscale('log')
ax_main[0].set(xlabel='age [Myr]')
ax_main[0].set_title('internal', fontsize=10)

# Plot second panel: internal + external photoevaporation
sns.histplot(data=data_external_plot, x="age", y="r1", bins=(100, 100),
             cbar=False, stat='density',
             cmap='turbo', norm=LogNorm(vmin=1.e-3, vmax=1.e-1),
             vmin=None, vmax=None, log_scale=(False, True),
             kde=True, ax=ax_main[1])
ax_main[1].set(xlabel='')

# Configure panel b
ax_main[1].set_ylabel("")
ax_main[1].tick_params(left=False, labelleft=False)
ax_main[1].set_xlim(0., 20.)
ax_main[1].set_ylim(r1_min, r1_max)
ax_main[1].set_yscale('log')
ax_main[1].set(xlabel='age [Myr]')
ax_main[1].set_title('internal + external', fontsize=10)

# ============================================================================
# KDE AT SPECIFIC AGES ANALYSIS
# ============================================================================
print("\nComputing KDE at specific ages...")

# Define ages of interest and time windows
ages_of_interest = [1.0, 5.0]  # Myr
age_window = 0.25  # ±0.5 Myr window around each age

# Extract initial r1 distribution from parameters.csv (same approach as fig1.py)
csv_path = paths.data / "parameters.csv"
print(f"Loading initial conditions from: {csv_path}")
df_params = pd.read_csv(csv_path)
r1_initial = df_params['r1'].values
r1_initial = r1_initial[(r1_initial >= r1_min) & (r1_initial <= r1_max)]
print(f"Initial r1 distribution: {len(r1_initial)} points from initial conditions")

# Use filtered data for KDE analysis
data_int_kde = data_internal_plot
data_ext_kde = data_external_plot

# Evaluation points (log-spaced for log scale)
r1_eval = np.logspace(np.log10(r1_min), np.log10(r1_max), 100)

# Create KDE for initial conditions (age-independent)
if len(r1_initial) > 1:
    kde_initial = gaussian_kde(r1_initial)
    kde_initial_vals = kde_initial(r1_eval)
else:
    print(f"WARNING: Not enough initial data for KDE (found {len(r1_initial)} points)")
    kde_initial_vals = np.zeros_like(r1_eval)

# Dictionary to store KDE values for each model and age
kde_results = {
    'internal': {},
    'external': {}
}

# Compute KDE for each age
for age_target in ages_of_interest:
    print(f"\nProcessing age = {age_target} Myr (window: ±{age_window} Myr)")
    
    # Filter internal data at this age
    age_mask_int = ((data_int_kde['age'] >= age_target - age_window) & 
                    (data_int_kde['age'] <= age_target + age_window))
    r1_int_age = data_int_kde['r1'][age_mask_int].values
    
    # Filter external data at this age
    age_mask_ext = ((data_ext_kde['age'] >= age_target - age_window) & 
                    (data_ext_kde['age'] <= age_target + age_window))
    r1_ext_age = data_ext_kde['r1'][age_mask_ext].values
    
    print(f"  Internal: {len(r1_int_age)} points")
    print(f"  External: {len(r1_ext_age)} points")
    
    # Create KDE for internal at this age
    if len(r1_int_age) > 1:
        kde_int = gaussian_kde(r1_int_age)
        kde_results['internal'][age_target] = kde_int(r1_eval)
    else:
        print(f"  WARNING: Not enough Internal data at {age_target} Myr")
        kde_results['internal'][age_target] = np.zeros_like(r1_eval)
    
    # Create KDE for external at this age
    if len(r1_ext_age) > 1:
        kde_ext = gaussian_kde(r1_ext_age)
        kde_results['external'][age_target] = kde_ext(r1_eval)
    else:
        print(f"  WARNING: Not enough External data at {age_target} Myr")
        kde_results['external'][age_target] = np.zeros_like(r1_eval)

# ============================================================================
# PLOT KDE AT SPECIFIC AGES
# ============================================================================
# Plot critical radius KDE at specific ages to the right of panel b
# Line styles for different ages
linestyles = {1.0: '-', 5.0: '--'}
linewidths = {1.0: 2.0, 5.0: 1.8}

# Plot initial conditions (age-independent, shown once)
ax_kde_r1.plot(kde_initial_vals, r1_eval, 'k:', linewidth=1.5,
               label='initial', alpha=0.7)

# Plot internal photoevaporation at different ages
for age in ages_of_interest:
    label = f'internal ({age:.0f} Myr)'
    ax_kde_r1.plot(kde_results['internal'][age], r1_eval, 
                   color='#0072B2', linestyle=linestyles[age], 
                   linewidth=linewidths[age], label=label, alpha=0.8)

# Plot internal + external photoevaporation at different ages
for age in ages_of_interest:
    label = f'internal+ext ({age:.0f} Myr)'
    ax_kde_r1.plot(kde_results['external'][age], r1_eval, 
                   color='#009E73', linestyle=linestyles[age], 
                   linewidth=linewidths[age], label=label, alpha=0.8)

# Add shaded region between models at 5 Myr (as an example)
ax_kde_r1.fill_betweenx(r1_eval, 
                        kde_results['internal'][5.0],
                        kde_results['external'][5.0], 
                        alpha=0.15, color='gray')

ax_kde_r1.set_xlabel('Density', fontsize=10)
ax_kde_r1.set_ylabel("$r_1$ [AU]")
ax_kde_r1.yaxis.set_label_position("right")
ax_kde_r1.yaxis.tick_right()
ax_kde_r1.set_ylim(r1_min, r1_max)
ax_kde_r1.set_xlim(0, None)  # Auto-scale for linear
ax_kde_r1.set_yscale('log')
ax_kde_r1.legend(loc='upper right', fontsize=5.5, framealpha=0.9)
ax_kde_r1.grid(True, alpha=0.3)

# Add subplot labels
ax_main[0].text(0.98, 0.98, 'a)', transform=ax_main[0].transAxes, fontsize=10,
                fontweight='bold', va='top', ha='right')
ax_main[1].text(0.98, 0.98, 'b)', transform=ax_main[1].transAxes, fontsize=10,
                fontweight='bold', va='top', ha='right')

# ============================================================================
# SAVE FIGURE
# ============================================================================
output_path = paths.figures / 'Fig7.png'
fig.savefig(output_path, format='png', bbox_inches='tight')
print(f"\nFigure saved as: {output_path}")

print("\nFig7 generation complete!")
