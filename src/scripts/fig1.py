"""
Script to create a corner-style plot of r1 vs disc mass with marginal KDE distributions.

Creates a main scatter plot of critical radius vs disc mass with KDE distributions
along the margins showing the individual parameter distributions.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import paths
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
import subprocess
import matplotlib.font_manager as font_manager

# Try to use science plots style if available
try:
    import scienceplots
    plt.style.use('science')
    print("Using scienceplots style for publication quality")
except ImportError:
    print("scienceplots not available, using default matplotlib style")

font_loc1 = "/usr/local/texlive/2022/texmf-dist/fonts/opentype/public/tex-gyre"
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

# Constants
AU = 1.496e13  # cm to AU conversion
M_sun = 1.989e33  # g


def collect_parameters(csv_path):
    """Collect all parameters from CSV file."""
    import pandas as pd
    
    data_list = []
    
    print(f"Reading parameters from {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        print(f"Found {len(df)} rows in CSV file")
        print(f"CSV columns: {list(df.columns)}")
        
        # Convert each row to a dictionary with the expected parameter names
        for idx, row in df.iterrows():
            params = {}
            
            # Map CSV columns to parameter names
            if 'mstar' in df.columns:
                params['mstar'] = float(row['mstar'])
            if 'L_x' in df.columns:
                params['L_x'] = float(row['L_x'])
            if 'm0' in df.columns:
                params['m0'] = float(row['m0'])
            if 'r1' in df.columns:
                params['r1'] = float(row['r1'])
            if 'alpha' in df.columns:
                params['alpha'] = float(row['alpha'])
            if 'Habing_G0' in df.columns:
                params['Habing_G0'] = float(row['Habing_G0'])
            
            data_list.append(params)
        
        # Print a sample of available parameters for debugging
        if len(data_list) > 0:
            print(f"Sample parameters from first data point: {list(data_list[0].keys())}")
            
            # Check for disc mass related parameters
            disc_params = [key for key in data_list[0].keys() if 'disc' in key.lower() or 'mass' in key.lower() or 'm0' in key.lower()]
            print(f"Disc/mass related parameters: {disc_params}")
            
            # Check for r1 related parameters  
            r1_params = [key for key in data_list[0].keys() if 'r1' in key.lower() or 'radius' in key.lower()]
            print(f"r1/radius related parameters: {r1_params}")
        
        print(f"Successfully read parameters from {len(data_list)} rows")
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    return data_list


def power_law(x, a, b):
    """Power law function: y = a * x^b"""
    return a * x**b


def create_mstar_lx_corner_plot(data_list, output_file='mstar_lx_corner.png'):
    """Create corner-style plot for stellar mass vs X-ray luminosity."""
    
    # Extract data
    mstar_values = []
    lx_values = []
    
    for params in data_list:
        if 'mstar' in params and 'L_x' in params:
            mstar_values.append(params['mstar'])
            lx_values.append(params['L_x'])
    
    if len(mstar_values) == 0:
        print("No valid mstar vs L_x data found!")
        return np.array([]), np.array([])
    
    mstar_values = np.array(mstar_values)
    lx_values = np.array(lx_values)
    
    print("Stellar Mass vs X-ray Luminosity corner plot analysis:")
    print(f"N = {len(mstar_values)}")
    print(f"M_star range = {mstar_values.min():.2f} - {mstar_values.max():.2f} M_sun")
    print(f"L_x range = {lx_values.min():.2f} - {lx_values.max():.2f} x 10^30 erg/s")
    
    # Create figure optimized for MNRAS single-column (10/3 inches wide)
    fig = plt.figure(figsize=(10/3, 3.2))
    
    # Create gridspec: main plot takes most space, margins for KDEs
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[3, 0.8, 0.2],
                  height_ratios=[0.6, 3, 0.6],
                  hspace=0.12, wspace=0.12)
    
    # Adjust margins for publication quality - space for y-labels
    fig.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.92)
    
    # Define subplots
    ax_main = fig.add_subplot(gs[1, 0])      # Main scatter plot
    ax_right = fig.add_subplot(gs[1, 1])     # Right KDE (L_x)
    ax_bottom = fig.add_subplot(gs[2, 0])    # Bottom KDE (M_star)
    
    # Main scatter plot (log-log) with publication-optimized settings
    ax_main.loglog(mstar_values, lx_values, 'o', alpha=0.7, markersize=2,
                   color='darkred', label='Data', markeredgewidth=0)
    
    # Add observed relation L_x ∝ M_*^1.54 (normalized to median of data)
    median_mstar = np.median(mstar_values)
    median_lx = np.median(lx_values)
    norm_factor = median_lx / (median_mstar**1.54)
    
    # Plot observed relation line
    mstar_fit = np.logspace(np.log10(mstar_values.min()),
                            np.log10(mstar_values.max()), 100)
    lx_obs = norm_factor * (mstar_fit**1.54)
    ax_main.loglog(mstar_fit, lx_obs, color='green', linewidth=1.5,
                   linestyle=':', label=r'$L_X \propto M_\star^{1.54}$')
    
    print("Added observed relation: L_x ∝ M_*^1.54")
    
    ax_main.set_xlabel('')  # Remove bottom xlabel
    ax_main.set_ylabel(r'$L_X$ [10$^{30}$ erg s$^{-1}$]')
    ax_main.yaxis.set_label_coords(-0.18, 0.5)  # Move y-label further left
    ax_main.legend(fontsize=7, loc='upper left', frameon=True,
                   fancybox=True, shadow=False)
    ax_main.grid(True, alpha=0.3, linewidth=0.5)
    
    # Move x-axis ticks and labels to the top
    ax_main.xaxis.tick_top()
    ax_main.xaxis.set_label_position('top')
    ax_main.set_xlabel(r'M$_*$ [M$_\odot$]')
    
    # Right panel: KDE of L_x values (vertical orientation) in log space
    lx_positive = lx_values[lx_values > 0]
    if len(lx_positive) > 0:
        log_lx = np.log10(lx_positive)
        kde_log_lx = gaussian_kde(log_lx)
        log_lx_range = np.linspace(log_lx.min(), log_lx.max(), 200)
        kde_log_lx_values = kde_log_lx(log_lx_range)
        
        # Convert back to linear space for plotting
        lx_range = 10**log_lx_range
        
        ax_right.plot(kde_log_lx_values, lx_range, 'r-', linewidth=2)
        ax_right.fill_betweenx(lx_range, 0, kde_log_lx_values,
                               alpha=0.3, color='red')
    else:
        print("Warning: No positive L_x values for log KDE")
    ax_right.set_yscale('log')
    ax_right.set_ylim(ax_main.get_ylim())  # Match main plot y-axis
    ax_right.set_xlabel('Density')
    ax_right.set_ylabel('')
    ax_right.yaxis.set_ticklabels([])  # Remove y-tick labels
    ax_right.grid(True, alpha=0.3)
    
    # Statistical labels removed for cleaner publication appearance
    
    # Bottom panel: KDE of stellar masses (horizontal orientation) in log space
    mstar_positive = mstar_values[mstar_values > 0]
    if len(mstar_positive) > 0:
        log_mstar = np.log10(mstar_positive)
        kde_log_mstar = gaussian_kde(log_mstar)
        log_mstar_range = np.linspace(log_mstar.min(), log_mstar.max(), 200)
        kde_log_mstar_values = kde_log_mstar(log_mstar_range)
        
        # Convert back to linear space for plotting
        mstar_range = 10**log_mstar_range
        
        ax_bottom.plot(mstar_range, kde_log_mstar_values, 'b-', linewidth=2)
        ax_bottom.fill_between(mstar_range, 0, kde_log_mstar_values, alpha=0.3,
                               color='blue')
    else:
        print("Warning: No positive M_star values for log KDE")
    ax_bottom.set_xscale('log')
    ax_bottom.set_xlim(ax_main.get_xlim())  # Match main plot x-axis
    ax_bottom.set_ylabel('Density')
    ax_bottom.set_xlabel('')
    ax_bottom.tick_params(axis='x', which='both', bottom=False, top=False,
                          labelbottom=False)  # Remove x-ticks and labels
    ax_bottom.grid(True, alpha=0.3)
    
    # Statistical labels removed for cleaner publication appearance
    
    # Save the figure
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"Stellar mass vs X-ray luminosity corner plot saved as: {output_file}")
    plt.close(fig)
    
    return mstar_values, lx_values


def create_corner_plot(data_list, output_file='corner_plot.png'):
    """Create corner-style plot for critical radius vs disc mass."""
    
    # Extract data
    r1_values = []
    mdisk_values = []
    
    for params in data_list:
        if 'r1' in params and 'm0' in params:
            r1_values.append(params['r1'])
            mdisk_values.append(params['m0'] * 1000)  # Convert to Earth masses
    
    if len(r1_values) == 0:
        print("No valid r1 vs disc_mass data found!")
        return np.array([]), np.array([])
    
    r1_values = np.array(r1_values)
    mdisk_values = np.array(mdisk_values)
    
    print("Critical Radius vs Disc Mass corner plot analysis:")
    print(f"N = {len(r1_values)}")
    print(f"r1 range = {r1_values.min():.2f} - {r1_values.max():.2f} AU")
    print(f"m0 range = {mdisk_values.min():.2f} - {mdisk_values.max():.2f} "
          f"M_earth")
    
    # Create figure optimized for MNRAS single-column (10/3 inches wide)
    fig = plt.figure(figsize=(10/3, 3.2))
    
    # Create gridspec: main plot takes most space, margins for KDEs
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[3, 0.8, 0.2],
                  height_ratios=[0.6, 3, 0.6],
                  hspace=0.12, wspace=0.12)
    
    # Adjust margins for publication quality - space for y-labels
    fig.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.92)
    
    # Define subplots
    ax_main = fig.add_subplot(gs[1, 0])      # Main scatter plot
    ax_right = fig.add_subplot(gs[1, 1])     # Right KDE (disc mass)
    ax_bottom = fig.add_subplot(gs[2, 0])    # Bottom KDE (r1)
    
    # Main scatter plot (log-log) with publication-optimized settings
    ax_main.loglog(r1_values, mdisk_values, 'o', alpha=0.7, markersize=2,
                   color='steelblue', label='Data', markeredgewidth=0)
    
    # Add observed relation M_disc ∝ r_1^1.7 (normalized to median of data)
    median_r1 = np.median(r1_values)
    median_m0 = np.median(mdisk_values)
    norm_factor = median_m0 / (median_r1**1.7)
    
    # Plot observed relation line
    r1_fit = np.logspace(np.log10(r1_values.min()),
                         np.log10(r1_values.max()), 100)
    m0_obs = norm_factor * (r1_fit**1.7)
    ax_main.loglog(r1_fit, m0_obs, color='green', linewidth=1.5,
                   linestyle=':', label=r'$M_\mathrm{d} \propto {r_1}^{1.7}$')
    
    print("Added observed relation: M_disc ∝ r_1^1.7")
    
    ax_main.set_xlabel('')  # Remove bottom xlabel
    ax_main.set_ylabel(r'M$_{d}$ [M$_\oplus$]')
    ax_main.yaxis.set_label_coords(-0.18, 0.5)  # Move y-label further left
    ax_main.legend(fontsize=7, loc='upper left', frameon=True,
                   fancybox=True, shadow=False)
    ax_main.grid(True, alpha=0.3, linewidth=0.5)
    
    # Move x-axis ticks and labels to the top
    ax_main.xaxis.tick_top()
    ax_main.xaxis.set_label_position('top')
    ax_main.set_xlabel(r'r$_1$ [AU]')
    
    # Right panel: KDE of disc masses (vertical orientation) in log space
    # Filter out zero values for log calculation
    m0_nonzero = mdisk_values[mdisk_values > 0]
    if len(m0_nonzero) > 0:
        log_m0 = np.log10(m0_nonzero)
        kde_log_m0 = gaussian_kde(log_m0)
        log_m0_range = np.linspace(log_m0.min(), log_m0.max(), 200)
        kde_log_m0_values = kde_log_m0(log_m0_range)
        
        # Convert back to linear space for plotting
        m0_range = 10**log_m0_range
        
        ax_right.plot(kde_log_m0_values, m0_range, 'r-', linewidth=2)
        ax_right.fill_betweenx(m0_range, 0, kde_log_m0_values,
                               alpha=0.3, color='red')
    else:
        print("Warning: No positive disc mass values for log KDE")
    ax_right.set_yscale('log')
    ax_right.set_ylim(ax_main.get_ylim())  # Match main plot y-axis
    ax_right.set_xlabel('Density')
    ax_right.set_ylabel('')
    ax_right.yaxis.set_ticklabels([])  # Remove y-tick labels
    ax_right.grid(True, alpha=0.3)
    
    # Statistical labels removed for cleaner publication appearance
    
    # Bottom panel: KDE of critical radii (horizontal orientation) in log space
    # All r1 values should be positive, but check anyway
    r1_positive = r1_values[r1_values > 0]
    if len(r1_positive) > 0:
        log_r1 = np.log10(r1_positive)
        kde_log_r1 = gaussian_kde(log_r1)
        log_r1_range = np.linspace(log_r1.min(), log_r1.max(), 200)
        kde_log_r1_values = kde_log_r1(log_r1_range)
        
        # Convert back to linear space for plotting
        r1_range = 10**log_r1_range
        
        ax_bottom.plot(r1_range, kde_log_r1_values, 'b-', linewidth=2)
        ax_bottom.fill_between(r1_range, 0, kde_log_r1_values, alpha=0.3,
                               color='blue')
    else:
        print("Warning: No positive r1 values for log KDE")
    ax_bottom.set_xscale('log')
    ax_bottom.set_xlim(ax_main.get_xlim())  # Match main plot x-axis
    ax_bottom.set_ylabel('Density')
    ax_bottom.set_xlabel('')
    ax_bottom.tick_params(axis='x', which='both', bottom=False, top=False,
                          labelbottom=False)  # Remove x-ticks and labels
    ax_bottom.grid(True, alpha=0.3)
    
    # Statistical labels removed for cleaner publication appearance
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved as: {output_file}")
    plt.close(fig)
    
    return r1_values, mdisk_values


def create_combined_corner_plot(data_list,
                                output_file='fig.png',
                                data_list_oor=None):
    """Create a single figure with two corner plots and a KDE distribution.
    
    The figure contains three panels:
    1. Left: Corner plot of stellar mass vs X-ray luminosity
    2. Middle: Corner plot of critical radius vs disc mass
    3. Right: KDE distribution of external FUV field (G0)
    """
    
    # Extract data for both plots
    mstar_values = []
    lx_values = []
    r1_values = []
    mdisk_values = []
    g0_values = []
    alpha_log_values = []
    
    for params in data_list:
        if 'mstar' in params and 'L_x' in params:
            mstar_values.append(params['mstar'])
            lx_values.append(params['L_x'])
        if 'r1' in params and 'm0' in params:
            r1_values.append(params['r1'])
            mdisk_values.append(params['m0'] * 1000)  # Convert to Earth masses
        if 'Habing_G0' in params:
            g0_values.append(params['Habing_G0'])
        if 'alpha' in params and params['alpha'] > 0:
            alpha_log_values.append(np.log10(params['alpha']))
    
    mstar_values = np.array(mstar_values)
    lx_values = np.array(lx_values)
    r1_values = np.array(r1_values)
    mdisk_values = np.array(mdisk_values)
    g0_values = np.array(g0_values)
    alpha_log_values = np.array(alpha_log_values)

    # Extract out-of-range data (r1 < 1 AU or r1 > 500 AU) if provided
    r1_oor = np.array([])
    mdisk_oor = np.array([])
    if data_list_oor:
        r1_oor_list, mdisk_oor_list = [], []
        for params in data_list_oor:
            if 'r1' in params and 'm0' in params:
                r1_oor_list.append(params['r1'])
                mdisk_oor_list.append(params['m0'] * 1000)
        r1_oor = np.array(r1_oor_list)
        mdisk_oor = np.array(mdisk_oor_list)
    
    print("Figure 1 analysis:")
    print(f"Stellar mass vs X-ray luminosity: N = {len(mstar_values)}")
    print(f"Critical radius vs disc mass: N = {len(r1_values)}")
    print(f"External FUV field (G0): N = {len(g0_values[g0_values > 0])}")
    
    # Create figure optimized for MNRAS two-column (20/3 inches wide)
    fig = plt.figure(figsize=(10., 3.2))
    
    # Create gridspec for three corner plots side by side
    gs_main = GridSpec(1, 3, figure=fig, wspace=0.30)
    
    # Adjust margins for publication quality - more space for y-labels
    fig.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.90)
    
    # === LEFT PANEL: Stellar Mass vs X-ray Luminosity ===
    gs_left = gs_main[0].subgridspec(3, 3,
                                     width_ratios=[3, 0.8, 0.2],
                                     height_ratios=[0.6, 3, 0.6],
                                     hspace=0.12, wspace=0.12)
    
    ax_left_main = fig.add_subplot(gs_left[1, 0])
    ax_left_right = fig.add_subplot(gs_left[1, 1])
    ax_left_bottom = fig.add_subplot(gs_left[2, 0])
    
    # Left main scatter plot
    ax_left_main.loglog(mstar_values, lx_values, 'o', alpha=0.7, markersize=2,
                        color='steelblue', label='Data', markeredgewidth=0)
    
    # Add observed relation L_x ∝ M_*^1.54
    median_mstar = np.median(mstar_values)
    median_lx = np.median(lx_values)
    norm_factor = median_lx / (median_mstar**1.54)
    
    mstar_fit = np.logspace(np.log10(mstar_values.min()),
                            np.log10(mstar_values.max()), 100)
    lx_obs = norm_factor * (mstar_fit**1.54)
    ax_left_main.loglog(mstar_fit, lx_obs, color='red', linewidth=1.5,
                        linestyle='--', label=r'$L_X \propto M_\star^{1.54}$')
    
    ax_left_main.set_xlabel('')
    ax_left_main.set_ylabel(r'$L_X$ [10$^{30}$ erg s$^{-1}$]')
    ax_left_main.yaxis.set_label_coords(-0.22, 0.5)
    ax_left_main.legend(fontsize=7, loc='lower right', frameon=True)
    ax_left_main.grid(True, alpha=0.3, linewidth=0.5)
    
    # Move x-axis to top
    ax_left_main.xaxis.tick_top()
    ax_left_main.xaxis.set_label_position('top')
    ax_left_main.set_xlabel(r'M$_*$ [M$_\odot$]')
    
    # Left right panel: KDE of L_x values
    lx_positive = lx_values[lx_values > 0]
    if len(lx_positive) > 0:
        log_lx = np.log10(lx_positive)
        kde_log_lx = gaussian_kde(log_lx)
        log_lx_range = np.linspace(log_lx.min(), log_lx.max(), 200)
        kde_log_lx_values = kde_log_lx(log_lx_range)
        lx_range = 10**log_lx_range
        
        ax_left_right.plot(kde_log_lx_values, lx_range, 'r-', linewidth=2)
        ax_left_right.fill_betweenx(lx_range, 0, kde_log_lx_values,
                                    alpha=0.3, color='red')
    
    ax_left_right.set_yscale('log')
    ax_left_right.set_ylim(ax_left_main.get_ylim())
    ax_left_right.set_xlabel('Density')
    ax_left_right.set_ylabel('')
    ax_left_right.yaxis.set_ticklabels([])
    ax_left_right.grid(True, alpha=0.3)
    
    # Left bottom panel: KDE of stellar masses
    mstar_positive = mstar_values[mstar_values > 0]
    if len(mstar_positive) > 0:
        log_mstar = np.log10(mstar_positive)
        kde_log_mstar = gaussian_kde(log_mstar)
        log_mstar_range = np.linspace(log_mstar.min(), log_mstar.max(), 200)
        kde_log_mstar_values = kde_log_mstar(log_mstar_range)
        mstar_range = 10**log_mstar_range
        
        ax_left_bottom.plot(mstar_range, kde_log_mstar_values, 'b-',
                            linewidth=2)
        ax_left_bottom.fill_between(mstar_range, 0, kde_log_mstar_values,
                                    alpha=0.3, color='blue')
    
    ax_left_bottom.set_xscale('log')
    ax_left_bottom.set_xlim(ax_left_main.get_xlim())
    ax_left_bottom.set_ylabel('Density')
    ax_left_bottom.set_xlabel('')
    ax_left_bottom.tick_params(axis='x', which='both', bottom=False, top=False,
                               labelbottom=False)
    ax_left_bottom.grid(True, alpha=0.3)
    
    # === RIGHT PANEL: Critical Radius vs Disc Mass ===
    gs_right = gs_main[1].subgridspec(3, 3,
                                      width_ratios=[3, 0.8, 0.2],
                                      height_ratios=[0.6, 3, 0.6],
                                      hspace=0.12, wspace=0.12)
    
    ax_right_main = fig.add_subplot(gs_right[1, 0])
    ax_right_right = fig.add_subplot(gs_right[1, 1])
    ax_right_bottom = fig.add_subplot(gs_right[2, 0])
    
    # Right main scatter plot: out-of-range behind, in-range on top
    if len(r1_oor) > 0:
        ax_right_main.loglog(r1_oor, mdisk_oor, 'o', alpha=0.3, markersize=2,
                             color='gray', markeredgewidth=0)
    ax_right_main.loglog(r1_values, mdisk_values, 'o', alpha=0.7, markersize=2,
                         color='steelblue', label='Data', markeredgewidth=0)
    
    # Add observed relation M_disc ∝ r_1^1.7
    median_r1 = np.median(r1_values)
    median_m0 = np.median(mdisk_values)
    norm_factor = median_m0 / (median_r1**1.7)
    
    r1_fit = np.logspace(np.log10(r1_values.min()),
                         np.log10(r1_values.max()), 100)
    m0_obs = norm_factor * (r1_fit**1.7)
    ax_right_main.loglog(r1_fit, m0_obs, color='red', linewidth=1.5,
                         linestyle='--',
                         label=r'$M_\mathrm{d} \propto {r_1}^{1.7}$')
    
    ax_right_main.set_xlabel('')
    ax_right_main.set_ylabel(r'M$_{d}$ [M$_\oplus$]')
    ax_right_main.yaxis.set_label_coords(-0.22, 0.5)
    ax_right_main.legend(fontsize=7, loc='lower right', frameon=True)
    ax_right_main.grid(True, alpha=0.3, linewidth=0.5)
    
    # Move x-axis to top
    ax_right_main.xaxis.tick_top()
    ax_right_main.xaxis.set_label_position('top')
    ax_right_main.set_xlabel(r'r$_1$ [AU]')

    # Dashed vertical lines marking the considered r1 range
    ax_right_main.axvline(1.0, color='k', linestyle='--', linewidth=0.8,
                          alpha=0.6)
    ax_right_main.axvline(500.0, color='k', linestyle='--', linewidth=0.8,
                          alpha=0.6)
    
    # Right right panel: KDE of disc masses
    m0_nonzero = mdisk_values[mdisk_values > 0]
    if len(m0_nonzero) > 0:
        log_m0 = np.log10(m0_nonzero)
        kde_log_m0 = gaussian_kde(log_m0)
        log_m0_range = np.linspace(log_m0.min(), log_m0.max(), 200)
        kde_log_m0_values = kde_log_m0(log_m0_range)
        m0_range = 10**log_m0_range
        
        ax_right_right.plot(kde_log_m0_values, m0_range, 'r-', linewidth=2)
        ax_right_right.fill_betweenx(m0_range, 0, kde_log_m0_values,
                                     alpha=0.3, color='red')
    
    ax_right_right.set_yscale('log')
    ax_right_right.set_ylim(ax_right_main.get_ylim())
    ax_right_right.set_xlabel('Density')
    ax_right_right.set_ylabel('')
    ax_right_right.yaxis.set_ticklabels([])
    ax_right_right.grid(True, alpha=0.3)
    
    # Right bottom panel: KDE of critical radii (all data, split shading)
    r1_positive = r1_values[r1_values > 0]
    r1_oor_positive = r1_oor[r1_oor > 0] if len(r1_oor) > 0 else np.array([])
    all_r1_positive = (np.concatenate([r1_positive, r1_oor_positive])
                       if len(r1_oor_positive) > 0 else r1_positive)
    if len(all_r1_positive) > 0:
        log_r1_all = np.log10(all_r1_positive)
        kde_log_r1 = gaussian_kde(log_r1_all)
        log_r1_range = np.linspace(log_r1_all.min(), log_r1_all.max(), 200)
        kde_log_r1_values = kde_log_r1(log_r1_range)
        r1_range = 10**log_r1_range

        in_range = (r1_range >= 1.0) & (r1_range <= 500.0)
        ax_right_bottom.plot(r1_range, kde_log_r1_values, 'b-', linewidth=2)
        ax_right_bottom.fill_between(r1_range, 0, kde_log_r1_values,
                                     where=in_range, alpha=0.4,
                                     color='steelblue', label='1–500 AU')
        ax_right_bottom.fill_between(r1_range, 0, kde_log_r1_values,
                                     where=~in_range, alpha=0.15,
                                     color='gray', label='Outside range')
        # Vertical dashed lines at r1 boundaries
        ax_right_bottom.axvline(1.0, color='k', linestyle='--',
                                linewidth=0.8, alpha=0.6)
        ax_right_bottom.axvline(500.0, color='k', linestyle='--',
                                linewidth=0.8, alpha=0.6)
    
    ax_right_bottom.set_xscale('log')
    ax_right_bottom.set_xlim(ax_right_main.get_xlim())
    ax_right_bottom.set_ylabel('Density')
    ax_right_bottom.set_xlabel('')
    ax_right_bottom.tick_params(axis='x', which='both', bottom=False,
                                top=False, labelbottom=False)
    ax_right_bottom.grid(True, alpha=0.3)
    
    # === THIRD PANEL: KDE of External FUV Field (G0) ===
    ax_g0 = fig.add_subplot(gs_main[2])
    
    # Filter positive G0 values and convert to log10
    g0_positive = g0_values[g0_values > 0]
    log_g0_all = np.log10(g0_positive)
    
    if len(log_g0_all) > 1:
        # Create KDE
        kde_g0 = gaussian_kde(log_g0_all)
        log_g0_range = np.linspace(log_g0_all.min(), log_g0_all.max(), 200)
        kde_g0_vals = kde_g0(log_g0_range)
        
        # Plot KDE
        ax_g0.plot(log_g0_range, kde_g0_vals, 'steelblue', linewidth=2)
        ax_g0.fill_between(log_g0_range, 0, kde_g0_vals, alpha=0.4, 
                          color='steelblue')
        
        # Add histogram for context
        ax_g0.hist(log_g0_all, bins=30, density=True, alpha=0.3, 
                   color='gray', edgecolor='black', linewidth=0.5)
    
    ax_g0.set_xlabel(r'$\log_{10}(G_0)$')
    ax_g0.set_ylabel('Density')
    ax_g0.grid(True, alpha=0.3, linewidth=0.5)
    ax_g0.yaxis.set_label_coords(-0.15, 0.5)
    
    # Add statistics text
    if len(log_g0_all) > 0:
        median_g0 = np.median(log_g0_all)
        mean_g0 = np.mean(log_g0_all)
        std_g0 = np.std(log_g0_all)
        stats_text = (f'median = {median_g0:.2f}\n'
                     f'mean = {mean_g0:.2f}\n'
                     f'std = {std_g0:.2f}')
        ax_g0.text(0.97, 0.97, stats_text, transform=ax_g0.transAxes,
                   fontsize=7, verticalalignment='top', 
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))

    # Save the figure
    plt.savefig(paths.figures / output_file, dpi=300, bbox_inches='tight')
    print(f"Combined figure saved as: {output_file}")
    plt.close(fig)
    
    return mstar_values, lx_values, r1_values, mdisk_values


def main():
    # Path to the CSV file with parameters
    csv_path = paths.data / "parameters.csv"
    
    print(f"Analyzing disc parameters from: {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} does not exist!")
        return
    
    # Collect all parameters from CSV file
    data_list = collect_parameters(csv_path)

    if len(data_list) == 0:
        print("No valid parameter data found!")
        return

    # Load out-of-range parameters if available
    oor_csv_path = paths.data / "parameters_out_of_range.csv"
    data_list_oor = []
    if os.path.exists(oor_csv_path):
        print(f"\nLoading out-of-range parameters from: {oor_csv_path}")
        data_list_oor = collect_parameters(oor_csv_path)
        print(f"Loaded {len(data_list_oor)} out-of-range data points")
    else:
        print(f"No out-of-range parameter file found at {oor_csv_path}")
        print("Run ic.py to generate it.")
    
    # Create combined figure
    print("\n" + "="*60)
    print("CREATING FIGURE 1")
    print("="*60)
    mstar_vals, lx_vals, r1_vals, m0_vals = create_combined_corner_plot(
        data_list, 'Fig1.png', data_list_oor=data_list_oor)
    
    print("Figure 1 completed successfully!")
    print("Final statistics:")
    print(f"- M_star vs L_x data points: {len(mstar_vals)}")
    print(f"- r1 vs disc mass data points: {len(r1_vals)}")
    print(f"- M_star range: {mstar_vals.min():.2f} - {mstar_vals.max():.2f} M_sun")
    print(f"- L_x range: {lx_vals.min():.2f} - {lx_vals.max():.2f} x 10^30 erg/s")
    print(f"- r1 range: {r1_vals.min():.2f} - {r1_vals.max():.2f} AU")
    print(f"- Disc mass range: {m0_vals.min():.2f} - {m0_vals.max():.2f} M_earth")
    
    print("\n" + "="*60)
    print("FIGURE 1 COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
