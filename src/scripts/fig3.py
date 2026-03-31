#!/usr/bin/env python3
"""
Script to create publication-quality mass loss rate comparison figure
for scientific letter paper. Figure size optimized for single column A4 (~3.5 inches).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import paths
import os
from astropy import units as u
from scipy.optimize import curve_fit
import subprocess
import matplotlib.font_manager as font_manager

# Use scienceplots for publication-quality styling
try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
    print("Using scienceplots styling: 'science' + 'ieee'")
except ImportError:
    print("Warning: scienceplots not found. Install with: pip install scienceplots")
    print("Falling back to basic matplotlib styling...")

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

def load_mass_loss_data():
    """Load mass loss rate data from all simulation directories"""
    
    # Define the directories and labels
    gaps = ['10', '50', '100', '200']
    labels = [r'$r_1 = 10$ au', r'$r_1 = 50$ au', 
              r'$r_1 = 100$ au', r'$r_1 = 200$ au']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional color scheme
    markers = ['o', 's', '^', 'D']  # Different markers for each simulation
    fillstyles = ['full', 'full', 'none', 'none']  # Mix of filled and hollow markers
    
    data_dict = {}
    
    for i, gap in enumerate(gaps):
        try:
            # Construct file path
            file_path = paths.data / f'mass_loss_fit_{gap}.txt'
            
            # Load the data
            data = np.loadtxt(file_path)
            
            # Extract columns: radius, normalized mass loss, unnormalized mass loss
            radius_au = data[:, 0]
            normalized_mdot = data[:, 1] 
            unnormalized_mdot = data[:, 2]
            
            # Store in dictionary
            data_dict[gap] = {
                'radius': radius_au,
                'normalized': normalized_mdot,
                'unnormalized': unnormalized_mdot,
                'label': labels[i],
                'color': colors[i],
                'marker': markers[i],
                'fillstyle': fillstyles[i]
            }
            
            print(f"Loaded {len(radius_au)} data points from {gap}")
            
        except FileNotFoundError:
            print(f"Warning: File not found for {gap}")
        except Exception as e:
            print(f"Error loading data from {gap}: {e}")
    
    return data_dict


def power_law_func(r, A, alpha):
    """Power law function: M_dot = A * r^alpha"""
    return A * r**alpha


def remove_outliers_by_radius(data_dict, sigma_threshold=10.0, radius_bin_size=0.1):
    """
    Remove outlier data points that are more than sigma_threshold away from 
    the mean value at each radius bin.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing simulation data
    sigma_threshold : float
        Number of standard deviations for outlier detection (default: 10.0)
    radius_bin_size : float
        Size of radius bins in log10 space for local statistics (default: 0.1)
    
    Returns:
    --------
    cleaned_data_dict : dict
        Dictionary with outliers removed
    outlier_stats : dict
        Statistics about removed outliers
    """
    
    print(f"Removing outliers using {sigma_threshold}-sigma clipping...")
    print(f"Radius bin size in log10 space: {radius_bin_size}")
    print("=" * 60)
    
    cleaned_data_dict = {}
    outlier_stats = {
        'total_original': 0,
        'total_cleaned': 0,
        'total_removed': 0,
        'by_simulation': {}
    }
    
    for sim_name, data in data_dict.items():
        # Convert to M☉/yr for consistency with notebook
        mdot_msun_per_yr = (data['unnormalized'] * u.g/u.s).to(u.Msun/u.yr).value
        
        # Get original data
        radius = data['radius'].copy()
        unnormalized_msun_yr = mdot_msun_per_yr.copy()
        
        # Filter out zero values first
        mask_nonzero = unnormalized_msun_yr > 0
        radius_nz = radius[mask_nonzero]
        mdot_nz = unnormalized_msun_yr[mask_nonzero]
        
        original_count = len(radius_nz)
        outlier_stats['total_original'] += original_count
        
        if len(radius_nz) == 0:
            print(f"⚠ Warning: No non-zero data points for {sim_name}")
            continue
        
        # Convert to log space for binning
        log_radius = np.log10(radius_nz)
        log_mdot = np.log10(mdot_nz)
        
        # Create radius bins
        log_r_min, log_r_max = log_radius.min(), log_radius.max()
        n_bins = max(5, int((log_r_max - log_r_min) / radius_bin_size))
        bin_edges = np.linspace(log_r_min, log_r_max, n_bins + 1)
        
        # Initialize mask for keeping points
        keep_mask = np.ones(len(radius_nz), dtype=bool)
        total_outliers = 0
        
        # Process each bin
        for i in range(len(bin_edges) - 1):
            # Find points in this radius bin
            bin_mask = (log_radius >= bin_edges[i]) & (log_radius < bin_edges[i + 1])
            
            if i == len(bin_edges) - 2:  # Include the last edge in the final bin
                bin_mask = (log_radius >= bin_edges[i]) & (log_radius <= bin_edges[i + 1])
            
            bin_indices = np.where(bin_mask)[0]
            
            if len(bin_indices) < 3:  # Need at least 3 points to calculate statistics
                continue
            
            # Calculate statistics for this bin
            bin_mdot = log_mdot[bin_indices]
            mean_mdot = np.mean(bin_mdot)
            std_mdot = np.std(bin_mdot, ddof=1)  # Sample standard deviation
            
            if std_mdot == 0:  # All values are identical
                continue
            
            # Find outliers in this bin
            z_scores = np.abs(bin_mdot - mean_mdot) / std_mdot
            outlier_mask = z_scores > sigma_threshold
            
            # Mark outliers for removal
            outlier_indices = bin_indices[outlier_mask]
            keep_mask[outlier_indices] = False
            total_outliers += len(outlier_indices)
            
            # Debug info for bins with outliers
            if len(outlier_indices) > 0:
                r_bin_center = 10**((bin_edges[i] + bin_edges[i + 1]) / 2)
                print(f"  {sim_name}: Bin centered at {r_bin_center:.1f} AU - "
                      f"removed {len(outlier_indices)}/{len(bin_indices)} points "
                      f"(max z-score: {z_scores.max():.1f})")
        
        # Apply the cleaning mask
        radius_cleaned = radius_nz[keep_mask]
        mdot_cleaned = mdot_nz[keep_mask]
        
        # Store cleaned data
        cleaned_data_dict[sim_name] = data.copy()  # Start with original data
        cleaned_data_dict[sim_name]['radius_cleaned'] = radius_cleaned
        cleaned_data_dict[sim_name]['unnormalized_msun_yr_cleaned'] = mdot_cleaned
        
        cleaned_count = len(radius_cleaned)
        removed_count = original_count - cleaned_count
        
        # Store statistics
        outlier_stats['by_simulation'][sim_name] = {
            'original': original_count,
            'cleaned': cleaned_count,
            'removed': removed_count,
            'removal_rate': removed_count / original_count * 100
        }
        
        outlier_stats['total_cleaned'] += cleaned_count
        outlier_stats['total_removed'] += removed_count
        
        print(f"✓ {sim_name}: {original_count} → {cleaned_count} points "
              f"({removed_count} removed, {removed_count/original_count*100:.1f}%)")
    
    # Print summary statistics
    print(f"\n" + "=" * 60)
    print("OUTLIER REMOVAL SUMMARY")
    print("=" * 60)
    print(f"Total original points: {outlier_stats['total_original']}")
    print(f"Total cleaned points: {outlier_stats['total_cleaned']}")
    print(f"Total removed points: {outlier_stats['total_removed']}")
    print(f"Overall removal rate: {outlier_stats['total_removed']/outlier_stats['total_original']*100:.1f}%")
    print(f"Sigma threshold used: {sigma_threshold}")
    
    return cleaned_data_dict, outlier_stats


def fit_power_law_regime(radii, mdot_values, regime_name, r_min=None, r_max=None):
    """Fit power law to a specific radius regime"""
    
    # Apply radius mask
    mask = np.ones(len(radii), dtype=bool)
    if r_min is not None:
        mask &= (radii >= r_min)
    if r_max is not None:
        mask &= (radii <= r_max)
    
    r_fit = radii[mask]
    mdot_fit = mdot_values[mask]
    
    # Remove any remaining zero or negative values
    positive_mask = mdot_fit > 0
    r_fit = r_fit[positive_mask]
    mdot_fit = mdot_fit[positive_mask]
    
    if len(r_fit) < 3:
        print(f"⚠ Warning: Not enough data points for {regime_name} fitting ({len(r_fit)} points)")
        return None, None, None, None, None
    
    try:
        # Initial guess for parameters [A, alpha]
        # A ~ typical mass loss rate, alpha ~ 0.5 (typical scaling)
        initial_guess = [mdot_fit.mean(), 0.5]
        
        # Fit the power law
        popt, pcov = curve_fit(power_law_func, r_fit, mdot_fit, 
                              p0=initial_guess, 
                              maxfev=2000)
        
        A_fit, alpha_fit = popt
        
        # Calculate uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        A_err, alpha_err = param_errors
        
        # Calculate R-squared
        mdot_pred = power_law_func(r_fit, A_fit, alpha_fit)
        ss_res = np.sum((mdot_fit - mdot_pred) ** 2)
        ss_tot = np.sum((mdot_fit - np.mean(mdot_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return A_fit, alpha_fit, A_err, alpha_err, r_squared
        
    except Exception as e:
        print(f"✗ Error fitting {regime_name}: {e}")
        return None, None, None, None, None


def perform_two_regime_fitting(data_dict, use_cleaned_data=False):
    """Perform two-regime power law fitting on combined data"""
    
    # Collect all data points from all simulations
    if use_cleaned_data:
        print("Collecting CLEANED data for two-regime power law fitting...")
    else:
        print("Collecting data for two-regime power law fitting...")
    print("=" * 60)
    
    all_r_data = []
    all_mdot_data = []
    
    for sim_name, data in data_dict.items():
        if use_cleaned_data and 'radius_cleaned' in data:
            # Use cleaned data
            r_data = data['radius_cleaned']
            mdot_data = data['unnormalized_msun_yr_cleaned']
            print(f"  {sim_name}: Using {len(r_data)} cleaned data points")
        else:
            # Convert to M☉/yr and filter out zero values
            mdot_msun_per_yr = (data['unnormalized'] * u.g/u.s).to(u.Msun/u.yr).value
            mask_nonzero = mdot_msun_per_yr > 0
            
            r_data = data['radius'][mask_nonzero]
            mdot_data = mdot_msun_per_yr[mask_nonzero]
            print(f"  {sim_name}: Using {len(r_data)} original data points")
        
        all_r_data.extend(r_data)
        all_mdot_data.extend(mdot_data)
    
    # Convert to numpy arrays
    all_r_data = np.array(all_r_data)
    all_mdot_data = np.array(all_mdot_data)
    
    print(f"Total data points: {len(all_r_data)}")
    print(f"Radius range: {all_r_data.min():.2f} - {all_r_data.max():.2f} AU")
    print(f"Mass loss range: {all_mdot_data.min():.2e} - {all_mdot_data.max():.2e} M☉/yr")
    
    # Define regimes
    inner_regime_limit = 11.0  # AU
    outer_regime_limit = 40.0  # AU
    
    print(f"\nRegime definitions:")
    print(f"  Inner regime: r < {inner_regime_limit} AU")
    print(f"  Outer regime: r > {outer_regime_limit} AU")
    
    # Fit inner regime (r < 11 AU)
    print(f"\nFitting inner regime (r < {inner_regime_limit} AU)...")
    inner_results = fit_power_law_regime(
        all_r_data, all_mdot_data, "inner regime", r_min=6, r_max=inner_regime_limit
    )
    
    # Fit outer regime (r > 40 AU)  
    print(f"\nFitting outer regime (r > {outer_regime_limit} AU)...")
    outer_results = fit_power_law_regime(
        all_r_data, all_mdot_data, "outer regime", r_min=outer_regime_limit
    )
    
    # Print results
    print(f"\n" + "=" * 60)
    print("TWO-REGIME POWER LAW FITTING RESULTS")
    print("=" * 60)
    
    if inner_results[1] is not None:
        A_inner, alpha_inner, A_inner_err, alpha_inner_err, r2_inner = inner_results
        inner_mask = all_r_data < inner_regime_limit
        inner_points = np.sum(inner_mask & (all_mdot_data > 0))
        
        print(f"\n📍 INNER REGIME (r < {inner_regime_limit} AU):")
        print(f"   Data points: {inner_points}")
        print(f"   Power law: Ṁ = {A_inner:.2e} × r^{alpha_inner:.3f}")
        print(f"   Exponent: α = {alpha_inner:.3f} ± {alpha_inner_err:.3f}")
        print(f"   Prefactor: A = {A_inner:.2e} ± {A_inner_err:.2e} M☉/yr")
        print(f"   R-squared: {r2_inner:.4f}")
        print(f"   Comparison with theory:")
        print(f"     α = {alpha_inner:.3f} vs α = 0.57 (diff: {alpha_inner - 0.57:+.3f})")
    
    if outer_results[1] is not None:
        A_outer, alpha_outer, A_outer_err, alpha_outer_err, r2_outer = outer_results
        outer_mask = all_r_data > outer_regime_limit
        outer_points = np.sum(outer_mask & (all_mdot_data > 0))
        
        print(f"\n📍 OUTER REGIME (r > {outer_regime_limit} AU):")
        print(f"   Data points: {outer_points}")
        print(f"   Power law: Ṁ = {A_outer:.2e} × r^{alpha_outer:.3f}")
        print(f"   Exponent: α = {alpha_outer:.3f} ± {alpha_outer_err:.3f}")
        print(f"   Prefactor: A = {A_outer:.2e} ± {A_outer_err:.2e} M☉/yr")
        print(f"   R-squared: {r2_outer:.4f}")
        print(f"   Comparison with theory:")
        print(f"     α = {alpha_outer:.3f} vs α = 0.57 (diff: {alpha_outer - 0.57:+.3f})")
    
    # Statistical comparison
    if inner_results[1] is not None and outer_results[1] is not None:
        print(f"\n📊 REGIME COMPARISON:")
        print(f"   Inner vs Outer exponent: {alpha_inner:.3f} vs {alpha_outer:.3f}")
        print(f"   Exponent difference: Δα = {alpha_outer - alpha_inner:+.3f}")
        print(f"   Exponent ratio: α_outer/α_inner = {alpha_outer/alpha_inner:.3f}")
    
    print(f"\n" + "=" * 60)
    
    return {
        'inner': inner_results,
        'outer': outer_results,
        'inner_limit': inner_regime_limit,
        'outer_limit': outer_regime_limit,
        'all_r': all_r_data,
        'all_mdot': all_mdot_data
    }


def create_publication_figure(data_dict, fitting_results=None, 
                            save_path='Fig2.png',
                            use_cleaned_data=False):
    """Create publication-quality mass loss rate comparison figure"""
    
    # Create figure with MNRAS single column width (10/3 inches)
    fig, ax = plt.subplots(1, 1, figsize=(10/3, 2.25))
    
    # Plot data for each simulation
    for directory, data in data_dict.items():
        if use_cleaned_data and 'radius_cleaned' in data:
            # Use cleaned data
            radius_filtered = data['radius_cleaned']
            mdot_filtered = data['unnormalized_msun_yr_cleaned']
            
            # Apply plot range filter
            mask_range = ((radius_filtered >= 5) & (radius_filtered <= 200) & 
                         (mdot_filtered >= 6e-10) & (mdot_filtered <= 6e-8))
            radius_filtered = radius_filtered[mask_range]
            mdot_filtered = mdot_filtered[mask_range]
        else:
            # Convert mass loss rate to solar masses per year
            mdot_msun_per_yr = (data['unnormalized'] * u.g/u.s).to(u.Msun/u.yr).value
            
            # Filter out zero values for log plotting
            mask_nonzero = mdot_msun_per_yr > 0
            radius_filtered = data['radius'][mask_nonzero]
            mdot_filtered = mdot_msun_per_yr[mask_nonzero]
        
        # Sort by radius to find the outermost point
        sort_indices = np.argsort(radius_filtered)
        radius_sorted = radius_filtered[sort_indices]
        mdot_sorted = mdot_filtered[sort_indices]
        
        # Plot all points except the last (outermost) one
        # Data points now more prominent than fits
        ax.loglog(radius_sorted[:-1], mdot_sorted[:-1], 
                 color=data['color'], 
                 marker=data['marker'],
                 fillstyle=data['fillstyle'],  # Mix of filled and hollow markers
                 linestyle='',  # No connecting lines, just markers
                 markersize=1.8,  # Smaller markers to reduce visual clutter
                 markeredgewidth=0.6,
                 markeredgecolor=data['color'],
                 label=data['label'],
                 alpha=0.6,  # Increased from 0.5 for more prominence
                 zorder=6)  # Higher zorder than fits
        
        # Highlight the outermost data point with full opacity and larger size
        ax.loglog(radius_sorted[-1:], mdot_sorted[-1:], 
                 color=data['color'], 
                 marker=data['marker'],
                 fillstyle='full',  # Always filled for emphasis
                 linestyle='',
                 markersize=4.5,  # Larger size for emphasis
                 markeredgewidth=1.2,  # Thicker edge
                 markeredgecolor='black',  # Dark edge for contrast
                 alpha=1.0,  # Full opacity for maximum emphasis
                 zorder=8)  # Highest zorder to always be visible
    
    # Add theoretical R^0.57 scaling for comparison
    # r_theory = np.logspace(0.7, 2.3, 100)  # 5 to 200 AU
    
    # Normalize to match data at 50 AU (use first available dataset)
    # reference_data = list(data_dict.values())[0]
    # reference_mdot = (reference_data['unnormalized'] * u.g/u.s).to(u.Msun/u.yr).value
    # ref_radius = 50.0
    # ref_idx = np.argmin(np.abs(reference_data['radius'] - ref_radius))
    # normalization = reference_mdot[ref_idx] / (ref_radius**0.57)
    
    # mdot_theory = normalization * r_theory**0.57
    
    # ax.loglog(r_theory, mdot_theory, 'k--', linewidth=1.5, alpha=0.7,
    #         label=r'$\dot{M} \propto r^{0.57}$', zorder=3)
    
    # Add two-regime fits if available
    if fitting_results is not None:
        inner_results = fitting_results['inner']
        outer_results = fitting_results['outer']
        inner_limit = fitting_results['inner_limit']
        outer_limit = fitting_results['outer_limit']
        
        # Plot inner regime fit (r < 11 AU) - extended with dashed gray lines
        if inner_results[1] is not None:
            A_inner, alpha_inner = inner_results[0], inner_results[1]
            
            # Extended range for visibility (from 5 AU to beyond fitting range)
            r_inner_extended = np.logspace(np.log10(5), np.log10(15), 100)
            mdot_inner_extended = power_law_func(r_inner_extended, A_inner,
                                                 alpha_inner)
            
            # Plot extended fit - more subdued to emphasize data
            ax.loglog(
                r_inner_extended,
                mdot_inner_extended,
                color='#6699DD',  # Lighter blue for subtlety
                linestyle='-',
                linewidth=2.0,  # Slightly thinner
                alpha=0.3,  # Much lower opacity to stay in background
                label=f'Inner: $R^{{{alpha_inner:.2f}}}$',
                zorder=3  # Lower z-order so data points are on top
            )
        
        # Plot outer regime fit (r > 40 AU) - extended with dashed gray lines
        if outer_results[1] is not None:
            A_outer, alpha_outer = outer_results[0], outer_results[1]
            
            # Extended range for visibility (from before fitting range)
            r_outer_extended = np.logspace(np.log10(15), np.log10(200), 100)
            mdot_outer_extended = power_law_func(r_outer_extended, A_outer,
                                                 alpha_outer)
            
            # Plot extended fit - more subdued to emphasize data
            ax.loglog(
                r_outer_extended,
                mdot_outer_extended,
                color='#DD6666',  # Lighter red for subtlety
                linestyle='-',
                linewidth=2.0,  # Slightly thinner
                alpha=0.3,  # Much lower opacity to stay in background
                label=f'Outer: $R^{{{alpha_outer:.2f}}}$',
                zorder=3  # Lower z-order so data points are on top
            )
        
        # Add vertical lines to show regime boundaries
        # ax.axvline(inner_limit, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
        # ax.axvline(outer_limit, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    
    # Set axis properties
    ax.set_xlabel(r'Radius [AU]', fontsize=SMALL_SIZE)
    ax.set_ylabel(r'$\dot{M}_{\mathrm{w}}$ [M$_{\odot}$ yr$^{-1}$]', fontsize=SMALL_SIZE)
    
    # Set axis limits as requested
    ax.set_xlim(6, 200)
    ax.set_ylim(8e-10, 4e-8)
    
    # Add subtle grid that doesn't compete with data/fits
    ax.grid(True, which='major', alpha=0.25, linewidth=0.5, color='gray', linestyle=':')
    ax.grid(True, which='minor', alpha=0.12, linewidth=0.3, color='gray', linestyle=':')
    
    # Add legend
    ax.legend(loc='lower right', fontsize=7, frameon=True, 
              fancybox=False, shadow=False, framealpha=0.95,
              edgecolor='black', facecolor='white')
    
    # Fine-tune tick parameters
    ax.tick_params(which='major', length=4, width=1.0)
    ax.tick_params(which='minor', length=2, width=0.8)
    ax.minorticks_on()
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save figure
    save_path = paths.figures / "Fig3.png"
    plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.05)
    
    print(f"Figure saved as {save_path}")
    
    return fig, ax

def print_statistics(data_dict):
    """Print statistics for each simulation"""
    
    print("\nSimulation Statistics:")
    print("=" * 60)
    
    for directory, data in data_dict.items():
        mdot_msun_per_yr = (data['unnormalized'] * u.g/u.s).to(u.Msun/u.yr).value
        
        print(f"\n{data['label']}:")
        print(f"  Radius range: {data['radius'].min():.1f} - {data['radius'].max():.1f} au")
        print(f"  Max mass loss rate: {mdot_msun_per_yr.max():.2e} M☉/yr")
        print(f"  Min mass loss rate: {mdot_msun_per_yr.min():.2e} M☉/yr")
        print(f"  Data points: {len(data['radius'])}")

def main():
    """Main function to create publication figure"""
    
    print("Creating publication-quality mass loss rate comparison figure...")
    print("=" * 60)
    
    # Load data
    data_dict = load_mass_loss_data()
    
    if not data_dict:
        print("Error: No data loaded. Please check file paths.")
        return
    
    # Print statistics
    print_statistics(data_dict)
    
    # Apply 10-sigma outlier removal
    print("\n" + "=" * 60)
    print("APPLYING 10-SIGMA OUTLIER REMOVAL")
    print("=" * 60)
    cleaned_data_dict, outlier_stats = remove_outliers_by_radius(data_dict, sigma_threshold=1.0)
    
    # Perform two-regime power law fitting on cleaned data
    print("\n" + "=" * 60)
    print("PERFORMING TWO-REGIME POWER LAW FITTING (CLEANED DATA)")
    print("=" * 60)
    fitting_results = perform_two_regime_fitting(cleaned_data_dict, use_cleaned_data=True)
    
    # Create figure with fitting results using cleaned data
    fig, ax = create_publication_figure(cleaned_data_dict, fitting_results, 
                                      use_cleaned_data=True)
    
    print("\nPublication figure created successfully!")
    print("Figure specifications:")
    print("  - Size: 3.5 × 2.8 inches (single column A4)")
    print("  - Resolution: 300 DPI")
    print("  - Format: PDF and PNG")
    print("  - Font: Professional serif fonts")
    print("  - Style: Publication-ready")
    print("  - Features: Two-regime power law fits (r < 12 AU, r > 40 AU)")
    print("  - Data: 10-sigma outlier removal applied")
    print(f"  - Outliers removed: {outlier_stats['total_removed']} ({outlier_stats['total_removed']/outlier_stats['total_original']*100:.1f}%)")
    print(f"  - Final data points: {outlier_stats['total_cleaned']}")

if __name__ == "__main__":
    main()
