"""
Publication-ready script to generate disc fraction vs age comparison figure.

This script creates a figure comparing population synthesis predictions with
observational data from Mamajek+2009,    # Add vertical line at e-folding time
    ax.vlines(tau_fit, 0, 90, color='red', ls=':', alpha=0.8, linewidth=2)
    ax.text(tau_fit + 0.3, 75, f'tau={tau_fit:.1f} Myr',
            rotation=0, fontsize=8, color='red', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.7))ng disc fraction evolution over time
for different photoevaporation prescriptions.

Author: Giovanni Picogna
Date: 10.05.2023
Last Updated: October 2025 - Polished for publication
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import paths
from scipy.optimize import curve_fit
from lib import load_data
from astropy import constants as const
from astropy import units as u
import os
import subprocess
import matplotlib.font_manager as font_manager
import glob
from functions import read_data, read_grid, remove_disc, calc_streamline, regrid, calculate_mdot_wind, save_dict


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
# Create 2x2 figure for different Rcrit directories with shared axes
# fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=False, sharey=False)
fig, axes = plt.subplots(2, 2, figsize=(20/3, 5.), sharex=False, sharey=False)
fig.subplots_adjust(left=.08, bottom=.08, right=.88, top=.94, hspace=0.08, wspace=0.08)

# Define directories and labels
base_path = paths.data / 'PLUTO_runs'
directories = ['10au', '50au', '100au', '200au']
labels = ['R$_{crit}$ = 10 AU', 'R$_{crit}$ = 50 AU', 'R$_{crit}$ = 100 AU', 'R$_{crit}$ = 200 AU']
xlims = [300, 300, 1000, 1000]
rcrits = [10, 50, 100, 200]  # Cut-off radii for each directory

# Stellar mass (assuming 1 Msun)
Mstar = 1.0 * const.M_sun

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Store plot object for shared colorbar
plot_obj = None

for idx, (directory, label, xl, rcrit) in enumerate(zip(directories, labels, xlims, rcrits)):
    ax = axes_flat[idx]
    
    # Change to directory and read outputs
    output_dir = os.path.join(base_path, directory, 'out')
    os.chdir(output_dir)
    
    # Get list of data files and select last 10
    data_files = sorted(glob.glob('data.*.dbl.h5'))
    last_10_files = data_files[-10:]
    
    print(f"Processing {directory}: {len(last_10_files)} files")
    
    # Initialize arrays for averaging
    density_avg = None
    vx_avg = None
    vz_avg = None
    column_density_avg = None
    
    # Read and average the last 10 outputs
    for file in last_10_files:
        grid_temp = read_grid(file)
        data_temp = read_data(file, grid_temp)
        
        if density_avg is None:
            density_avg = data_temp["density"]
            vx_avg = data_temp["v_x"]
            vz_avg = data_temp["v_z"]
            column_density_avg = data_temp["column_density"]
        else:
            density_avg += data_temp["density"]
            vx_avg += data_temp["v_x"]
            vz_avg += data_temp["v_z"]
            column_density_avg += data_temp["column_density"]
    
    # Calculate averages
    n_files = len(last_10_files)
    density_avg /= n_files
    vx_avg /= n_files
    vz_avg /= n_files
    column_density_avg /= n_files
    
    # Convert to appropriate units
    X_AU = (grid_temp["X"]*u.cm).to(u.AU).value
    Z_AU = (grid_temp["Z"]*u.cm).to(u.AU).value
    vx_kms = (vx_avg*u.cm/u.s).to(u.km/u.s).value
    vz_kms = (vz_avg*u.cm/u.s).to(u.km/u.s).value
    
    # Calculate gravitationally unbound region
    # Total velocity magnitude
    v_total = np.sqrt(vx_avg**2 + vz_avg**2)  # in cm/s
    
    # Distance from star (r = sqrt(x^2 + z^2))
    R_grid = grid_temp["X"]  # in cm
    Z_grid = grid_temp["Z"]  # in cm
    r_distance = np.sqrt(R_grid**2 + Z_grid**2)  # in cm
    
    # Escape velocity: v_esc = sqrt(2*G*M/r)
    v_escape = np.sqrt(2 * const.G.cgs.value * Mstar.cgs.value / r_distance)

    # Strategy: Mask out the spherical outer boundary where flow artificially vanishes
    # The boundary appears as an arc because it's a spherical shell at constant radius
    
    # Use density as primary indicator - outer boundary has very low artificial density
    density_threshold = 1e-22  # g/cm^3 - adjust based on your data
    density_mask = density_avg < density_threshold
    
    # Additionally detect regions where both velocity magnitude is very small
    # This catches the boundary layer where flow is artificially damped
    # v_threshold_high = 1e5  # 10 km/s - must have at least this velocity to be "active"
    # velocity_mask = v_total > v_threshold_high
    
    # Combine: need sufficient density OR sufficient velocity to be physical
    # This allows the unbound region to extend far out where there's real flow
    outer_boundary = density_mask # | velocity_mask
    
    # Unbound region: where v_total > v_escape AND in physical domain
    unbound_mask = (v_total > v_escape) | outer_boundary
    
    # Find contour of unbound region
    unbound_contour_value = 1.0
    
    # Downsample for cleaner vector display
    skip = 20
    
    # Set plot properties
    ax.set_xlim(0, xl)
    ax.set_ylim(0, xl)
    ax.set_aspect('equal')
    
    # Add labels only to left and bottom subplots
    if idx in [0, 2]:  # Left column
        ax.set_ylabel('Z [AU]')
    if idx in [2, 3]:  # Bottom row
        ax.set_xlabel('R [AU]')
    
    # Remove y-axis ticks and labels for right column
    if idx in [1, 3]:  # Right column
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
    # Plot density
    valmin = -20
    valmax = -14
    labelplot = '$\\log_{10}(\\rho)$ [$10^{-24}$ g cm$^{-3}$]'
    plot = ax.pcolormesh(X_AU, Z_AU, np.log10(density_avg),
                         vmin=valmin, vmax=valmax, cmap=plt.cm.jet)
    
    # Store plot object for colorbar (use last one)
    plot_obj = plot
    
    # Add contours for column density
    # vals = [5.e20, 5.e21]
    #circle = plt.Circle((0, 0), ((grid_temp['r'][0,fin]*u.cm).to(u.au)).value, 
    #                    color='k', ls='dashed', fill=False, linewidth=1.5)
    #ax.add_patch(circle)
    # contours1 = ax.contour(X_AU, Z_AU, column_density_avg, vals, 
    #                        linestyles='dashed', colors='red', linewidths=1.5)
    
    # Add contour for gravitationally unbound region
    unbound_contour = ax.contour(X_AU, Z_AU, unbound_mask.astype(float), 
                                 levels=[0.5], linestyles='dashed', 
                                 colors='red', linewidths=2.0)
    
    # Add velocity vectors
    Q = ax.quiver(X_AU[::skip, ::skip], Z_AU[::skip, ::skip], 
                  vx_kms[::skip, ::skip], vz_kms[::skip, ::skip],
                  color='white', alpha=0.6, scale=300, width=0.003)
    
    # Add velocity scale legend
    ax.quiverkey(Q, 0.85, 0.95, 25, '25 km/s', labelpos='E', 
                 coordinates='axes', color='red', labelcolor='red')
    # Clean up spines and ticks
    ax.tick_params(which='both', top=False, bottom=True, left=True, right=False, 
                   labelleft=True, labelbottom=True, direction='out')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

# Add single colorbar on the right side
cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.86])  # [left, bottom, width, height]
cbar = fig.colorbar(plot_obj, cax=cbar_ax, orientation='vertical')
cbar.set_label(labelplot, labelpad=15)

# Save figure with high resolution for publication
output_filename = paths.figures / 'Fig2.png'
fig.savefig(output_filename, format='png', bbox_inches='tight',
            facecolor='white', edgecolor='none')
