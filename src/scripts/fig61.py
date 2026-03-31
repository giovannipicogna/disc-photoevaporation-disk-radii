"""
Publication-ready script to generate Figure 6: Box plots and KDE difference analysis.

This script creates a comprehensive figure comparing photoevaporation prescriptions
through:
- 2D histograms of population synthesis results (age vs accretion rate)
- Box plots of observational data grouped by star-forming regions
- KDE marginal distributions showing prescription differences

The figure layout consists of:
- Panel a: Compact with observational box plots
- Panel b: Compact PE/10 (critical radius dependence) with observational box plots
- Panel c: Extended discs with observational box plots
- Bottom: Age-marginalized KDE differences
- Right: Accretion rate-marginalized KDE differences

Author: Giovanni Picogna
Date: 08.05.2023 - Modified with box plots and KDE difference plots
Last Updated: October 2025 - Polished for publication
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
from lib import load_data
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
    'figure.figsize': [20./3., 4.0],  # MNRAS single column width
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# ============================================================================
# FIGURE SETUP AND LAYOUT
# ============================================================================
# Create main figure optimized for MNRAS two-column layout (20/3 inches wide)
fig = plt.figure(figsize=(20./3., 4.))

# Create sophisticated gridspec layout:
# - Column 0: Dedicated colorbar (width_ratio=0.15, smaller for compact layout)
# - Columns 1-3: Main plots (width_ratio=2 each for equal panels)
# - Column 4: KDE mdot plot (width_ratio=1.0, reduced for compactness)
# - Row 0: Main plots (height_ratio=1.0)
# - Row 1: KDE age plot (height_ratio=0.5, slightly reduced)
gs = gridspec.GridSpec(2, 5,
                       width_ratios=[0.15, 2, 2, 2, 1.],
                       height_ratios=[1.0, 0.5],
                       left=0.06, bottom=0.12, right=0.98, top=0.95,
                       hspace=0.08, wspace=0.05)

# Define subplot axes with shared axes for consistency
cbar_ax_models = fig.add_subplot(gs[0, 0])  # Dedicated colorbar axis
ax_kde_age = fig.add_subplot(gs[1, 3])      # KDE age plot (bottom, under panel c)

# Main comparison panels with shared x-axis
ax_main = [
    fig.add_subplot(gs[0, 1], sharex=ax_kde_age),  # Panel a: Compact
    fig.add_subplot(gs[0, 2], sharex=ax_kde_age),  # Panel b: Compact PE/10
    fig.add_subplot(gs[0, 3], sharex=ax_kde_age)   # Panel c: Extended
]

# KDE mdot plot shares y-axis with main panels
ax_kde_mdot = fig.add_subplot(gs[0, 4], sharey=ax_main[0])

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
# Define data paths
data_path = "/e/ocean1/users/picogna/Photoevaporation-TransitionDiscs-Project/"

# Data filtering parameters
mask = False          # No additional masking applied
mask_val = 1.e-13     # Threshold value for potential masking

# Load population synthesis data for the new prescription
path = paths.data / 'pop_compact_discs/'
profile_name = "Compact discs"
data_compact = load_data(path, profile_name=profile_name, mask=mask,
                     mask_val=mask_val)
# data_compact["age"] += 1.  # Add 1 Myr to avoid log(0) issues

# Load population synthesis data for new prescription with reduced PE rates
path = paths.data / 'pop_compact_discs_reduced/'
profile_name = "Compact discs w PE/10"
data_compact_PE10 = load_data(path, profile_name=profile_name, mask=mask,
                      mask_val=mask_val)
# data_compact_PE10["age"] += 1.  # Add 1 Myr to avoid log(0) issues

# Load population synthesis data for compact discs (outer radius project)
path = paths.data / 'pop_compact_discs_old/'
profile_name = "Extended discs"
data_extended = load_data(path, profile_name=profile_name, mask=mask,
                         mask_val=mask_val)

# Print diagnostic information about loaded data
print(f"\nData loaded:")
print(f"Compact data points: {len(data_compact)}")
print(f"Compact PE/10 data points: {len(data_compact_PE10)}")
print(f"Extended data points: {len(data_extended)}")
if len(data_compact) > 0:
    print(f"Compact age range: {data_compact['age'].min():.2f} - {data_compact['age'].max():.2f} Myr")
    print(f"Compact mdot range: {data_compact['mdot_acc'].min():.2e} - {data_compact['mdot_acc'].max():.2e} M_sun/yr")
else:
    print("WARNING: Compact dataset is empty!")
if len(data_compact_PE10) > 0:
    print(f"Compact PE/10 age range: {data_compact_PE10['age'].min():.2f} - {data_compact_PE10['age'].max():.2f} Myr")
    print(f"Compact PE/10 mdot range: {data_compact_PE10['mdot_acc'].min():.2e} - {data_compact_PE10['mdot_acc'].max():.2e} M_sun/yr")
else:
    print("WARNING: Compact PE/10 dataset is empty!")
if len(data_extended) > 0:
    print(f"Extended age range: {data_extended['age'].min():.2f} - {data_extended['age'].max():.2f} Myr")
    print(f"Extended mdot range: {data_extended['mdot_acc'].min():.2e} - {data_extended['mdot_acc'].max():.2e} M_sun/yr")
else:
    print("WARNING: Extended dataset is empty!")

# ============================================================================
# OBSERVATIONAL DATA LOADING AND CLEANING
# ============================================================================
# Load observational accretion rate data from PPVII compilation
acc_data = pd.read_csv(paths.data / 'data-PPVII.csv', sep='\t')

# Clean column names and region values (remove whitespace)
acc_data.columns = acc_data.columns.str.strip()
acc_data['Region'] = acc_data['Region'].str.strip()

# Convert string columns to numeric, handling invalid entries gracefully
acc_data['Mstar_PPVII'] = pd.to_numeric(acc_data['Mstar_PPVII'],
                                        errors='coerce')
acc_data['logMacc_PPVII'] = pd.to_numeric(acc_data['logMacc_PPVII'],
                                          errors='coerce')

# Apply quality filters to remove invalid/placeholder values
valid_data_mask = ((acc_data['Mstar_PPVII'] > 0) &
                   (acc_data['logMacc_PPVII'] > -90) &
                   (acc_data['Mstar_PPVII'].notna()) &
                   (acc_data['logMacc_PPVII'].notna()))
acc_data = acc_data[valid_data_mask]

# Convert log accretion rates to linear values
acc_data["Macc"] = 10**(acc_data["logMacc_PPVII"])

# ============================================================================
# STAR-FORMING REGION AGE MAPPING
# ============================================================================
# Define representative ages and uncertainties for each star-forming region
# Based on literature compilation of cluster ages
age_mapping = {
    'Lupus': {'Age': 2, 'dAge': 1},      # Lupus complex
    'USco': {'Age': 8, 'dAge': 3},       # Upper Scorpius association  
    'ChamI': {'Age': 2.0, 'dAge': 0.5},  # Chamaeleon I
    'rOph': {'Age': 4, 'dAge': 2},       # rho Ophiuchus
    'ChamII': {'Age': 2, 'dAge': 0.5},   # Chamaeleon II
    'Taurus': {'Age': 1.5, 'dAge': 0.5}, # Taurus-Auriga
    'CrA': {'Age': 3, 'dAge': 1}         # Corona Australis
}

# Add Age and dAge columns based on Region
acc_data['Age'] = acc_data['Region'].map(
    lambda x: age_mapping.get(x, {}).get('Age', np.nan))
acc_data['dAge'] = acc_data['Region'].map(
    lambda x: age_mapping.get(x, {}).get('dAge', np.nan))

# Debug: Print mapping results
print(f"Rows before age mapping: {len(acc_data)}")
print(f"Rows with valid ages: {acc_data['Age'].notna().sum()}")
print(f"Unique regions after mapping: {acc_data['Region'].unique()}")

# Remove rows where Age couldn't be mapped (unknown regions)
acc_data = acc_data.dropna(subset=['Age', 'dAge'])
print(f"Rows after removing NaN ages: {len(acc_data)}")

# Read and merge low accretors data
low_acc_data = pd.read_csv(paths.data / 'low_accretors.dat', sep=r'\s+')
print(f"Low accretors data shape: {low_acc_data.shape}")

# Check for duplicate sources between datasets
low_acc_sources = set(low_acc_data['name'])
ppvii_sources = set(acc_data['Source'])
duplicates = low_acc_sources.intersection(ppvii_sources)

print(f"Sources in low_accretors: {len(low_acc_sources)}")
print(f"Sources in PPVII data: {len(ppvii_sources)}")
print(f"Duplicate sources found: {len(duplicates)}")

if duplicates:
    print("Duplicate sources:")
    for dup in sorted(duplicates):
        print(f"  {dup}")
    print("\nRemoving duplicates from low_accretors data...")
    # Remove duplicates from low_acc_data to avoid double-counting
    low_acc_data = low_acc_data[~low_acc_data['name'].isin(duplicates)]
    print(f"Low accretors after removing duplicates: {len(low_acc_data)}")
else:
    print("No duplicate sources found - safe to merge!")

# Map low_accretors region names to match data-PPVII naming convention
region_name_mapping = {
    'Chamaeleon_I': 'ChamI',
    'Chamaelon_I': 'ChamI',  # Alternative spelling/misspelling
    'Upper_Sco': 'USco',
    'Lupus': 'Lupus',
    'Taurus': 'Taurus',
    'rho_Oph': 'rOph',
    'Chamaeleon_II': 'ChamII',
    'Corona_Australis': 'CrA'
}

# First, try to map known regions
low_acc_data['Region_mapped'] = low_acc_data.iloc[:, 1].map(region_name_mapping)

# For unmapped regions, use the original name (create new SFRs)
unmapped_mask = low_acc_data['Region_mapped'].isna()
low_acc_data.loc[unmapped_mask, 'Region_mapped'] = low_acc_data.loc[unmapped_mask].iloc[:, 1]

print("Low accretors region mapping:")
# Show mapped regions
for orig, mapped in region_name_mapping.items():
    count = (low_acc_data.iloc[:, 1] == orig).sum()
    if count > 0:
        print(f"  {orig} -> {mapped}: {count} sources")

# Show new SFRs (unmapped regions)
unmapped_regions = low_acc_data[unmapped_mask].iloc[:, 1].unique()
if len(unmapped_regions) > 0:
    print("  New SFRs from low accretors:")
    for region in unmapped_regions:
        count = (low_acc_data.iloc[:, 1] == region).sum()
        print(f"    {region}: {count} sources")

# Convert low_accretors data to match acc_data format
low_acc_converted = pd.DataFrame({
    'Region': low_acc_data['Region_mapped'],  # Use mapped/original region names
    'Source': low_acc_data['name'],
    'Age': low_acc_data['t'],  # Column 3: age in Myr (from low_acc data)
    'dAge': low_acc_data['dt'],  # Column 4: age error (from low_acc data)
    'Mstar_PPVII': low_acc_data['Mstar'],  # Column 5: stellar mass
    'logMacc_PPVII': np.log10(low_acc_data['Mdot'] * 1e-10),  # Column 6
    'Macc': low_acc_data['Mdot'] * 1e-10,  # Convert units
    'Macc_upper': (low_acc_data['Mdot'] + low_acc_data['dMdot1']) * 1e-10,
    'Macc_lower': (low_acc_data['Mdot'] - low_acc_data['dMdot2']) * 1e-10
})

print(f"Low accretors total sources included: {len(low_acc_converted)}")

# For regions that match the main dataset, use the standard age mapping
# For new regions, use the ages directly from the low_accretors data
matched_regions = low_acc_converted['Region'].isin(age_mapping.keys())

# Use standard age mapping for known regions
low_acc_converted.loc[matched_regions, 'Age'] = low_acc_converted.loc[matched_regions, 'Region'].map(
    lambda x: age_mapping.get(x, {}).get('Age', np.nan))
low_acc_converted.loc[matched_regions, 'dAge'] = low_acc_converted.loc[matched_regions, 'Region'].map(
    lambda x: age_mapping.get(x, {}).get('dAge', np.nan))

# For new regions, the Age and dAge are already set from the low_acc_data
print("Age assignment summary:")
print(f"  Matched to existing SFRs: {matched_regions.sum()} sources")
print(f"  New SFRs with individual ages: {(~matched_regions).sum()} sources")

print(f"Low accretors converted shape: {low_acc_converted.shape}")

# Merge the datasets
acc_data_merged = pd.concat([acc_data, low_acc_converted], ignore_index=True)
print(f"Merged data shape: {acc_data_merged.shape}")

# Use merged data for plotting
acc_data = acc_data_merged
print(f"Final acc_data shape: {len(acc_data)}")

acc_data = acc_data.rename(columns={'Mstar_PPVII': 'M_star'})

# ============================================================================
# BOX PLOT DATA PREPARATION FUNCTIONS
# ============================================================================

def create_boxplot_data(acc_data):
    """
    Prepare observational data for box plots grouped by star-forming regions.
    
    Parameters:
    -----------
    acc_data : pandas.DataFrame
        Combined observational data with Region, Age, and Macc columns
    
    Returns:
    --------
    tuple : (age_centers, boxplot_data, region_labels)
        - age_centers: list of representative ages for positioning
        - boxplot_data: list of arrays containing accretion rates per region
        - region_labels: list of formatted labels for each region
    """
    regions = acc_data['Region'].unique()
    
    age_bin_centers = []
    boxplot_data = []
    region_labels = []
    
    for region in regions:
        region_data = acc_data[acc_data['Region'] == region]
        if len(region_data) > 0:
            # Get representative age for this region
            age = region_data['Age'].iloc[0]
            macc_values = region_data['Macc'].values
            
            age_bin_centers.append(age)
            boxplot_data.append(macc_values)
            region_labels.append(f"{region}\n({age:.1f} Myr)")
    
    return age_bin_centers, boxplot_data, region_labels


def add_boxplots(ax_obj, age_centers, box_data, box_labels):
    """
    Add publication-quality box plots to the specified axes.
    
    Parameters:
    -----------
    ax_obj : matplotlib.axes.Axes
        The axes object to plot on
    age_centers : list
        X-positions for box plot placement
    box_data : list
        Data arrays for each box plot
    box_labels : list
        Labels for each box plot (not used directly but kept for consistency)
    
    Returns:
    --------
    dict : Box plot objects returned by matplotlib.boxplot
    """
    bp = ax_obj.boxplot(box_data, positions=age_centers, widths=0.8,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='red', linewidth=2),
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5))
    return bp


# ============================================================================
# PLOTTING EXECUTION
# ============================================================================
# Prepare box plot data from combined observational dataset
age_centers, box_data, box_labels = create_boxplot_data(acc_data)
print(f"Created box plot data for {len(box_data)} regions")

# Plot first panel: compact discs with standard PE rates
sns.histplot(data=data_compact, x="age", y="mdot_acc",
             binwidth=(0.1*20./6., 0.1), cbar=False, stat='density',
             cmap='turbo', norm=LogNorm(vmin=1.e-4, vmax=1.e-1),
             vmin=None, vmax=None, log_scale=(False, True),
             cbar_kws={'label': 'density'}, kde=True, ax=ax_main[0])

# Add box plots over the histogram
add_boxplots(ax_main[0], age_centers, box_data, box_labels)

ax_main[0].set_yscale('log')
# Remove y-axis label and tick labels from Panel a since they will be on KDE Mdot panel
ax_main[0].set_ylabel("")
ax_main[0].tick_params(left=False, labelleft=False)
ax_main[0].set_xlim(0., 20.)
ax_main[0].set_ylim(1e-13, 1e-6)
ax_main[0].set(xlabel='age [Myr]')

# Set evenly spaced x-axis ticks for Panel a
x_bin_width = 0.1*20.  # Same as histogram binwidth
x_ticks = np.arange(0, 20.1, x_bin_width)
ax_main[0].set_xticks(x_ticks)
# Set cleaner tick labels - show every other tick to avoid crowding
x_tick_labels = [f'{x:.1f}' if i % 10 == 0 else ''
                 for i, x in enumerate(x_ticks)]
ax_main[0].set_xticklabels(x_tick_labels)
ax_main[0].set_title('Compact discs', fontsize=10)

# Plot second panel: compact discs with reduced PE rates
sns.histplot(data=data_compact_PE10, x="age", y="mdot_acc", bins=(100, 100),
             # binwidth=(0.1*20./6., 0.1),
             cbar=True, stat='density',
             cmap='turbo', norm=LogNorm(vmin=1.e-4, vmax=1.e-1),
             vmin=None, vmax=None, log_scale=(False, True),
             cbar_kws={'orientation': 'vertical'},
             cbar_ax=cbar_ax_models,
             kde=True, ax=ax_main[1])
ax_main[1].set(xlabel='')

# Configure the colorbar properly with title and ticks on the left
cbar_ax_models.set_ylabel('Model density', rotation=90, fontsize=10)
cbar_ax_models.yaxis.set_label_position('left')
cbar_ax_models.yaxis.set_ticks_position('left')
cbar_ax_models.tick_params(labelsize=6, which='both', left=True, right=False)

# Add box plots over the histogram
add_boxplots(ax_main[1], age_centers, box_data, box_labels)

ax_main[1].set_yscale('log')
# Remove y-axis label and tick labels for Panel b since it shares with Panel a
ax_main[1].set_ylabel("")
ax_main[1].tick_params(left=False, labelleft=False)
# Remove x-axis label from Panel b since it shares x-axis with KDE age plot below
ax_main[1].tick_params(bottom=False, labelbottom=False)
ax_main[1].set_xlim(0., 20.)
ax_main[1].set_ylim(1e-13, 1e-6)
ax_main[1].set_title('Compact discs PE/10', fontsize=10)

# Plot third panel: extended discs
sns.histplot(data=data_extended, x="age", y="mdot_acc", bins=(100, 100),
             cbar=False, stat='density',
             cmap='turbo', norm=LogNorm(vmin=1.e-4, vmax=1.e-1),
             vmin=None, vmax=None, log_scale=(False, True),
             kde=True, ax=ax_main[2])
ax_main[2].set(xlabel='')

# Add box plots over the histogram
add_boxplots(ax_main[2], age_centers, box_data, box_labels)

ax_main[2].set_yscale('log')
# Remove y-axis label and tick labels for Panel c
ax_main[2].set_ylabel("")
ax_main[2].tick_params(left=False, labelleft=False)
# Remove x-axis label from Panel c since it shares x-axis with KDE age plot below
ax_main[2].tick_params(bottom=False, labelbottom=False)
ax_main[2].set_xlim(0., 20.)
ax_main[2].set_ylim(1e-13, 1e-6)
ax_main[2].set_title('Extended discs', fontsize=10)

# Compute KDE differences for the subplot panels
print("Computing KDE differences...")

# Create age bins for KDE comparison
age_bins = np.arange(0, 20.1, 0.25)  # Broader bins for KDE

# Calculate marginal distributions
# Age-marginalized (sum over all mdot values)
compact_age_dist = []
compact_PE10_age_dist = []
extended_age_dist = []

for i in range(len(age_bins)-1):
    age_mask_compact = (data_compact['age'] >= age_bins[i]) & (data_compact['age'] < age_bins[i+1])
    age_mask_compact_PE10 = (data_compact_PE10['age'] >= age_bins[i]) & (data_compact_PE10['age'] < age_bins[i+1])
    age_mask_extended = (data_extended['age'] >= age_bins[i]) & (data_extended['age'] < age_bins[i+1])
    
    compact_age_dist.append(np.sum(age_mask_compact))
    compact_PE10_age_dist.append(np.sum(age_mask_compact_PE10))
    extended_age_dist.append(np.sum(age_mask_extended))

# Normalize to get densities (with safety checks for empty data)
compact_age_dist = np.array(compact_age_dist)
compact_PE10_age_dist = np.array(compact_PE10_age_dist)
extended_age_dist = np.array(extended_age_dist)

if np.sum(compact_age_dist) > 0:
    compact_age_dist = compact_age_dist / np.sum(compact_age_dist)
else:
    print("WARNING: No Compact age distribution data found!")
    compact_age_dist = np.zeros_like(compact_age_dist)

if np.sum(compact_PE10_age_dist) > 0:
    compact_PE10_age_dist = compact_PE10_age_dist / np.sum(compact_PE10_age_dist)
else:
    print("WARNING: No Compact PE/10 age distribution data found!")
    compact_PE10_age_dist = np.zeros_like(compact_PE10_age_dist)
    
if np.sum(extended_age_dist) > 0:
    extended_age_dist = extended_age_dist / np.sum(extended_age_dist)
else:
    print("WARNING: No Extended age distribution data found!")
    extended_age_dist = np.zeros_like(extended_age_dist)

# Age bin centers for plotting
age_bin_centers = (age_bins[:-1] + age_bins[1:]) / 2

# Plot mdot-marginalized difference in the bottom subplot
mdot_mask_compact = (data_compact['mdot_acc'] >= 1e-13) & (data_compact['mdot_acc'] <= 1e-6)
mdot_mask_compact_PE10 = (data_compact_PE10['mdot_acc'] >= 1e-13) & (data_compact_PE10['mdot_acc'] <= 1e-6)
mdot_mask_extended = (data_extended['mdot_acc'] >= 1e-13) & (data_extended['mdot_acc'] <= 1e-6)

# Use KDE for smooth distributions

# Log-transform mdot data for KDE
log_mdot_compact = np.log10(data_compact['mdot_acc'][mdot_mask_compact])
log_mdot_compact_PE10 = np.log10(data_compact_PE10['mdot_acc'][mdot_mask_compact_PE10])
log_mdot_extended = np.log10(data_extended['mdot_acc'][mdot_mask_extended])

# Evaluation points
log_mdot_eval = np.linspace(-13, -6, 100)
mdot_eval = 10**log_mdot_eval

# Create KDE with safety checks for sufficient data
if len(log_mdot_compact) > 1:
    kde_compact = gaussian_kde(log_mdot_compact)
    kde_compact_vals = kde_compact(log_mdot_eval)
else:
    print(f"WARNING: Not enough Compact data for KDE (found {len(log_mdot_compact)} points), using zeros")
    kde_compact_vals = np.zeros_like(log_mdot_eval)

if len(log_mdot_compact_PE10) > 1:
    kde_compact_PE10 = gaussian_kde(log_mdot_compact_PE10)
    kde_compact_PE10_vals = kde_compact_PE10(log_mdot_eval)
else:
    print(f"WARNING: Not enough Compact PE/10 data for KDE (found {len(log_mdot_compact_PE10)} points), using zeros")
    kde_compact_PE10_vals = np.zeros_like(log_mdot_eval)

if len(log_mdot_extended) > 1:
    kde_extended = gaussian_kde(log_mdot_extended)
    kde_extended_vals = kde_extended(log_mdot_eval)
else:
    print(f"WARNING: Not enough Extended data for KDE (found {len(log_mdot_extended)} points), using zeros")
    kde_extended_vals = np.zeros_like(log_mdot_eval)

# Plot age distribution KDE differences below the right plot
ax_kde_age.plot(age_bin_centers, compact_age_dist, 'b-', linewidth=2, 
                label='Compact', alpha=0.8)
ax_kde_age.plot(age_bin_centers, compact_PE10_age_dist, 'r-', linewidth=2, 
                label='Compact PE/10', alpha=0.8)
ax_kde_age.plot(age_bin_centers, extended_age_dist, 'g-', linewidth=2, 
                label='Extended', alpha=0.8)
ax_kde_age.fill_between(age_bin_centers, 
                        np.minimum(np.minimum(compact_age_dist, compact_PE10_age_dist), extended_age_dist),
                        np.maximum(np.maximum(compact_age_dist, compact_PE10_age_dist), extended_age_dist), 
                        alpha=0.2, color='gray', label='Spread')

ax_kde_age.set_xlabel('age [Myr]')
ax_kde_age.set_ylabel('Density')
ax_kde_age.set_xlim(0., 20.)
# Set the same evenly spaced x-ticks as Panel a and b
ax_kde_age.set_xticks(x_ticks)
ax_kde_age.set_xticklabels(x_tick_labels)
ax_kde_age.legend(loc='upper right', fontsize=6)
# Remove grid from KDE age plot

# Plot accretion rate KDE differences to the right of the right plot (rotated)
ax_kde_mdot.plot(kde_compact_vals, mdot_eval, 'b-', linewidth=2, 
                 label='Compact', alpha=0.8)
ax_kde_mdot.plot(kde_compact_PE10_vals, mdot_eval, 'r-', linewidth=2, 
                 label='Compact PE/10', alpha=0.8)
ax_kde_mdot.plot(kde_extended_vals, mdot_eval, 'g-', linewidth=2, 
                 label='Extended', alpha=0.8)
ax_kde_mdot.fill_betweenx(mdot_eval, 
                          np.minimum(np.minimum(kde_compact_vals, kde_compact_PE10_vals), kde_extended_vals),
                          np.maximum(np.maximum(kde_compact_vals, kde_compact_PE10_vals), kde_extended_vals), 
                          alpha=0.2, color='gray', label='Spread')

ax_kde_mdot.set_xlabel('Density', fontsize=10)
# Move shared y-axis label to the right side of KDE Mdot panel
ax_kde_mdot.set_ylabel("$\\log_{10}(\\dot{M}_\\mathrm{acc} [M_{\\odot}\\,\\mathrm{yr}^{-1}])$")
ax_kde_mdot.yaxis.set_label_position("right")
ax_kde_mdot.yaxis.tick_right()
# Set y-limits to match the shared log scale
ax_kde_mdot.set_ylim(1e-13, 1e-6)
ax_kde_mdot.legend(loc='upper left', fontsize=6)
ax_kde_mdot.grid(True, alpha=0.3)

# Add subplot indices only for main panels
ax_main[0].text(0.98, 0.98, 'a)', transform=ax_main[0].transAxes, fontsize=10,
                fontweight='bold', va='top', ha='right')
ax_main[1].text(0.98, 0.98, 'b)', transform=ax_main[1].transAxes, fontsize=10,
                fontweight='bold', va='top', ha='right')
ax_main[2].text(0.98, 0.98, 'c)', transform=ax_main[2].transAxes, fontsize=10,
                fontweight='bold', va='top', ha='right')
# KDE plots are supporting plots - no panel labels needed

# Save figure
fig.savefig(paths.figures / 'Fig6.png', format='png', bbox_inches='tight')
print("Figure saved as: Fig6_boxplot_kde_large_visc.png")
