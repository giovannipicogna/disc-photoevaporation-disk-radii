#!/usr/bin/env python3
"""
Analyze the relationship between dust disc masses and stellar masses
from PP7-Manara.csv data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import paths

# Set matplotlib parameters for vector output
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

def perform_fit_and_plot(df, dust_mass_col, stellar_mass_col, output_suffix='', title_suffix='', add_ansdell_fit=False):
    """
    Perform linear fit in log-log space and create plot.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe with disc data
    dust_mass_col : str
        Column name for dust mass
    stellar_mass_col : str
        Column name for stellar mass
    output_suffix : str
        Suffix for output filenames
    title_suffix : str
        Suffix for plot title
    add_ansdell_fit : bool, optional
        If True, overplot the Ansdell et al. 2017 Lupus fit
    """
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {title_suffix}")
    print(f"{'='*60}")
    print(f"Total rows in this sample: {len(df)}")
    
    # Extract relevant columns and remove rows with missing or invalid data
    data = df[[stellar_mass_col, dust_mass_col]].copy()
    
    # Remove rows with NaN or non-numeric values
    data = data.replace(['nan', 'NaN', '--', ''], np.nan)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    
    # Remove -99 (missing data flag) and unphysical values
    # For young stars, stellar mass should be 0 < M* < 10 Msun
    data = data[(data[stellar_mass_col] > 0) & 
                (data[stellar_mass_col] < 10) & 
                (data[stellar_mass_col] != -99) &
                (data[dust_mass_col] > 0)]
    
    print(f"Valid data points after cleaning: {len(data)}")
    
    # Extract arrays
    mstar = data[stellar_mass_col].values  # Solar masses
    mdust = data[dust_mass_col].values  # Earth masses
    
    # Convert to log space
    log_mstar = np.log10(mstar)
    log_mdust = np.log10(mdust)
    
    # Perform linear regression in log-log space
    slope, intercept, r_value, p_value, std_err_slope = linregress(log_mstar, log_mdust)
    
    # Calculate standard error for the intercept
    n = len(log_mstar)
    x_mean = np.mean(log_mstar)
    
    # Residual standard error
    y_pred = slope * log_mstar + intercept
    residuals = log_mdust - y_pred
    s_res = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    # Standard error of intercept
    std_err_intercept = s_res * np.sqrt(1/n + x_mean**2 / np.sum((log_mstar - x_mean)**2))
    
    print("\n" + "="*60)
    print("LINEAR FIT IN LOG-LOG SPACE:")
    print("="*60)
    print(f"log10(Mdust) = ({slope:.3f} ± {std_err_slope:.3f}) * log10(Mstar) + ({intercept:.3f} ± {std_err_intercept:.3f})")
    print(f"\nAlternatively:")
    print(f"Mdust [M_Earth] = {10**intercept:.3f} * (Mstar [M_sun])^{slope:.3f}")
    print(f"\nFit parameters:")
    print(f"  Slope = {slope:.6f} ± {std_err_slope:.6f}")
    print(f"  Intercept = {intercept:.6f} ± {std_err_intercept:.6f}")
    print(f"  Dispersion = {s_res:.6f}")
    print(f"\nFit quality:")
    print(f"  R^2 = {r_value**2:.4f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Number of points = {len(mstar)}")
    print("="*60 + "\n")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the data points
    ax.scatter(log_mstar, log_mdust, alpha=0.6, s=50, 
               edgecolors='black', linewidths=0.5, 
               label=f'Data (N={len(mstar)})')
    
    # Plot the fit line with dispersion band
    log_mstar_fit = np.linspace(log_mstar.min(), log_mstar.max(), 100)
    log_mdust_fit = slope * log_mstar_fit + intercept
    ax.plot(log_mstar_fit, log_mdust_fit, 'r-', linewidth=2,
            label=f'Fit: $\\beta$={slope:.2f}±{std_err_slope:.2f}, $R^2$={r_value**2:.3f}')
    
    # Add dispersion band (±1σ) for our fit
    ax.fill_between(log_mstar_fit, 
                    log_mdust_fit - s_res,
                    log_mdust_fit + s_res,
                    alpha=0.2, color='red', 
                    label=f'This work: $\\delta$={s_res:.2f}')
    
    # Add Ansdell et al. 2017 Lupus fit if requested
    if add_ansdell_fit:
        # Ansdell et al. 2017: log(Mdust[M⊕]) = (1.2 ± 0.2) + (1.8 ± 0.4) log(M⋆[M⊙])
        # with dispersion δ = 0.9±0.1
        ansdell_slope = 1.8
        ansdell_intercept = 1.2
        ansdell_dispersion = 0.9
        
        # Calculate Ansdell fit line
        log_mdust_ansdell = ansdell_slope * log_mstar_fit + ansdell_intercept
        
        # Plot Ansdell fit line
        ax.plot(log_mstar_fit, log_mdust_ansdell, 'b--', linewidth=2,
                label=f'Ansdell+2017 Lupus: slope={ansdell_slope:.1f}, int={ansdell_intercept:.1f}')
        
        # Add dispersion band (±1σ)
        ax.fill_between(log_mstar_fit, 
                        log_mdust_ansdell - ansdell_dispersion,
                        log_mdust_ansdell + ansdell_dispersion,
                        alpha=0.2, color='blue', 
                        label=f'Ansdell+2017 dispersion ($\\delta$={ansdell_dispersion:.1f})')
    
    # Labels and formatting
    ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(M_{\rm dust} / M_\oplus)$', fontsize=14)
    ax.set_title(f'Dust Disc Mass vs Stellar Mass ({title_suffix})', fontsize=14)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add fit equation to plot
    fit_text = f'$\\alpha = {intercept:.2f} \\pm {std_err_intercept:.2f}$\n$\\beta = {slope:.2f} \\pm {std_err_slope:.2f}$\n$\\delta = {s_res:.2f}$'
    ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = paths.figures / f'mdust_vs_mstar{output_suffix}.eps'
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    # Also save as PNG for easy viewing
    output_path_png = paths.figures / f'mdust_vs_mstar{output_suffix}.png'
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path_png}")
    
    plt.close()
    
    # Save results to a text file
    results_path = paths.data / f'mdust_mstar_fit_results{output_suffix}.txt'
    with open(results_path, 'w') as f:
        f.write(f"Dust Disc Mass vs Stellar Mass Analysis - {title_suffix}\n")
        f.write("="*60 + "\n")
        f.write(f"Data source: PP7-Manara.csv\n")
        f.write(f"Sample: {title_suffix}\n")
        f.write(f"Stellar mass column: {stellar_mass_col}\n")
        f.write(f"Dust mass column: {dust_mass_col}\n")
        f.write(f"Number of valid data points: {len(mstar)}\n\n")
        f.write("Linear fit in log-log space:\n")
        f.write(f"  log10(Mdust) = ({slope:.6f} ± {std_err_slope:.6f}) * log10(Mstar) + ({intercept:.6f} ± {std_err_intercept:.6f})\n\n")
        f.write(f"Fit parameters:\n")
        f.write(f"  Slope = {slope:.6f} ± {std_err_slope:.6f}\n")
        f.write(f"  Intercept = {intercept:.6f} ± {std_err_intercept:.6f}\n")
        f.write(f"  Dispersion (δ) = {s_res:.6f}\n\n")
        f.write(f"Alternative form:\n")
        f.write(f"  Mdust [M_Earth] = {10**intercept:.6f} * (Mstar [M_sun])^{slope:.6f}\n\n")
        f.write(f"Fit quality:\n")
        f.write(f"  R^2 = {r_value**2:.6f}\n")
        f.write(f"  Correlation coefficient (R) = {r_value:.6f}\n")
        f.write(f"  p-value = {p_value:.6e}\n")
        f.write(f"  Residual standard error = {s_res:.6f}\n")
    
    print(f"Results saved to: {results_path}")
    print()


def fit_region(data, dust_mass_col, stellar_mass_col):
    """
    Fit a single region's data.
    
    Returns fit parameters or None if insufficient data.
    """
    # Extract relevant columns and remove rows with missing or invalid data
    data_clean = data[[stellar_mass_col, dust_mass_col]].copy()
    
    # Remove rows with NaN or non-numeric values
    data_clean = data_clean.replace(['nan', 'NaN', '--', ''], np.nan)
    data_clean = data_clean.apply(pd.to_numeric, errors='coerce')
    data_clean = data_clean.dropna()
    
    # Remove -99 (missing data flag) and unphysical values
    # For young stars, stellar mass should be 0 < M* < 10 Msun
    data_clean = data_clean[(data_clean[stellar_mass_col] > 0) & 
                            (data_clean[stellar_mass_col] < 10) & 
                            (data_clean[stellar_mass_col] != -99) &
                            (data_clean[dust_mass_col] > 0)]
    
    if len(data_clean) < 10:  # Need minimum points for meaningful fit
        return None
    
    # Extract arrays
    mstar = data_clean[stellar_mass_col].values
    mdust = data_clean[dust_mass_col].values
    
    # Convert to log space
    log_mstar = np.log10(mstar)
    log_mdust = np.log10(mdust)
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err_slope = linregress(log_mstar, log_mdust)
    
    # Calculate standard error for the intercept
    n = len(log_mstar)
    x_mean = np.mean(log_mstar)
    y_pred = slope * log_mstar + intercept
    residuals = log_mdust - y_pred
    s_res = np.sqrt(np.sum(residuals**2) / (n - 2))
    std_err_intercept = s_res * np.sqrt(1/n + x_mean**2 / np.sum((log_mstar - x_mean)**2))
    
    return {
        'log_mstar': log_mstar,
        'log_mdust': log_mdust,
        'slope': slope,
        'intercept': intercept,
        'std_err_slope': std_err_slope,
        'std_err_intercept': std_err_intercept,
        'dispersion': s_res,  # Residual standard deviation (intrinsic scatter)
        'r_value': r_value,
        'p_value': p_value,
        'n_points': n
    }


def plot_all_regions(df, dust_mass_col, stellar_mass_col):
    """Create multipanel plot with all star-forming regions."""
    
    # Ansdell et al. 2017 fits: log(Mdust[M⊕]) = α + β log(M⋆[M⊙]) with dispersion δ
    ansdell_fits = {
        'Taurus': {'alpha': 1.2, 'alpha_err': 0.1, 'beta': 1.7, 'beta_err': 0.2, 'delta': 0.7, 'delta_err': 0.1},
        'Lupus': {'alpha': 1.2, 'alpha_err': 0.2, 'beta': 1.8, 'beta_err': 0.4, 'delta': 0.9, 'delta_err': 0.1},
        'ChamI': {'alpha': 1.0, 'alpha_err': 0.1, 'beta': 1.8, 'beta_err': 0.3, 'delta': 0.8, 'delta_err': 0.1},
        'USco': {'alpha': 0.8, 'alpha_err': 0.2, 'beta': 2.4, 'beta_err': 0.4, 'delta': 0.7, 'delta_err': 0.1}
    }
    
    # Get unique regions sorted by number of sources
    region_counts = df['Region'].value_counts()
    regions = region_counts.index.tolist()
    
    print("\n" + "="*80)
    print("ANALYZING INDIVIDUAL STAR-FORMING REGIONS")
    print("="*80)
    
    # Create multipanel figure
    n_regions = len(regions)
    n_cols = 3
    n_rows = (n_regions + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_regions > 1 else [axes]
    
    # Store results for summary table
    results_table = []
    
    # Process each region
    for idx, region in enumerate(regions):
        ax = axes[idx]
        
        # Filter data for this region
        region_data = df[df['Region'] == region].copy()
        
        print(f"\n{region}:")
        print(f"  Total sources: {len(region_data)}")
        
        # Fit the data
        fit_result = fit_region(region_data, dust_mass_col, stellar_mass_col)
        
        if fit_result is None:
            print(f"  Insufficient data for fit (< 10 valid points)")
            ax.text(0.5, 0.5, f'{region}\nInsufficient data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12)
            ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$')
            ax.set_ylabel(r'$\log_{10}(M_{\rm dust} / M_\oplus)$')
            continue
        
        # Extract fit results
        log_mstar = fit_result['log_mstar']
        log_mdust = fit_result['log_mdust']
        slope = fit_result['slope']
        intercept = fit_result['intercept']
        std_err_slope = fit_result['std_err_slope']
        std_err_intercept = fit_result['std_err_intercept']
        dispersion = fit_result['dispersion']
        r_value = fit_result['r_value']
        p_value = fit_result['p_value']
        n_points = fit_result['n_points']
        
        print(f"  Valid points: {n_points}")
        print(f"  Slope: {slope:.3f} ± {std_err_slope:.3f}")
        print(f"  Intercept: {intercept:.3f} ± {std_err_intercept:.3f}")
        print(f"  Dispersion: {dispersion:.3f}")
        print(f"  R²: {r_value**2:.4f}")
        print(f"  p-value: {p_value:.3e}")
        
        # Store for summary table
        results_table.append({
            'Region': region,
            'N': n_points,
            'Slope': slope,
            'Slope_err': std_err_slope,
            'Intercept': intercept,
            'Intercept_err': std_err_intercept,
            'Dispersion': dispersion,
            'R2': r_value**2,
            'p_value': p_value
        })
        
        # Plot data points
        ax.scatter(log_mstar, log_mdust, alpha=0.6, s=30,
                  edgecolors='black', linewidths=0.5, label='Data')
        
        # Plot fit line with dispersion band
        log_mstar_fit = np.linspace(log_mstar.min(), log_mstar.max(), 100)
        log_mdust_fit = slope * log_mstar_fit + intercept
        ax.plot(log_mstar_fit, log_mdust_fit, 'r-', linewidth=2, 
                label=f'This work: $\\beta$={slope:.2f}')
        
        # Add dispersion band (±1σ) for our fit
        ax.fill_between(log_mstar_fit, 
                        log_mdust_fit - dispersion,
                        log_mdust_fit + dispersion,
                        alpha=0.2, color='red', 
                        label=f'$\\delta$={dispersion:.2f}')
        
        # Add Ansdell et al. 2017 fit if available for this region
        if region in ansdell_fits:
            ansdell = ansdell_fits[region]
            alpha_ans = ansdell['alpha']
            beta_ans = ansdell['beta']
            delta_ans = ansdell['delta']
            
            # Calculate Ansdell fit line
            log_mdust_ansdell = beta_ans * log_mstar_fit + alpha_ans
            
            # Plot Ansdell fit line and dispersion
            ax.plot(log_mstar_fit, log_mdust_ansdell, 'b--', linewidth=2,
                    label=f'Ansdell+17: $\\beta$={beta_ans:.1f}')
            ax.fill_between(log_mstar_fit,
                           log_mdust_ansdell - delta_ans,
                           log_mdust_ansdell + delta_ans,
                           alpha=0.15, color='blue',
                           label=f'$\\delta$={delta_ans:.1f}')
        
        # Formatting
        ax.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$', fontsize=11)
        ax.set_ylabel(r'$\log_{10}(M_{\rm dust} / M_\oplus)$', fontsize=11)
        ax.set_title(f'{region} (N={n_points})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=7, framealpha=0.9)
        
        # Add fit info to panel
        fit_text = f'$\\alpha = {intercept:.2f} \\pm {std_err_intercept:.2f}$\n$\\beta = {slope:.2f} \\pm {std_err_slope:.2f}$\n$\\delta = {dispersion:.2f}$\n$R^2 = {r_value**2:.3f}$'
        ax.text(0.05, 0.05, fit_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(n_regions, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save multipanel figure
    output_path = paths.figures / 'mdust_vs_mstar_regions.eps'
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"\nMultipanel figure saved to: {output_path}")
    
    output_path_png = paths.figures / 'mdust_vs_mstar_regions.png'
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"Multipanel figure saved to: {output_path_png}")
    
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE OF FITS BY REGION")
    print("="*80)
    print(f"{'Region':<12} {'N':>5} {'Slope':>10} {'Intercept':>10} {'Disp.':>6} {'R²':>8} {'p-value':>10}")
    print("-"*80)
    for result in results_table:
        print(f"{result['Region']:<12} {result['N']:>5} "
              f"{result['Slope']:>6.3f}±{result['Slope_err']:<.3f} "
              f"{result['Intercept']:>6.3f}±{result['Intercept_err']:<.3f} "
              f"{result['Dispersion']:>6.3f} "
              f"{result['R2']:>8.4f} {result['p_value']:>10.3e}")
    print("="*80)
    
    # Save summary table to file
    results_df = pd.DataFrame(results_table)
    results_path = paths.data / 'mdust_mstar_fits_by_region.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nSummary table saved to: {results_path}\n")
    
    return results_table


def plot_cumulative_dust_mass(df, dust_mass_col):
    """
    Create cumulative dust mass distribution plot for each star-forming region.
    Similar to dust mass survival curves.
    """
    
    # Get unique regions
    region_counts = df['Region'].value_counts()
    regions = region_counts.index.tolist()
    
    print("\n" + "="*80)
    print("CREATING CUMULATIVE DUST MASS DISTRIBUTIONS")
    print("="*80)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color scheme for regions
    colors = {
        'rOph': '#FF6B6B',      # red
        'Taurus': '#9B59B6',    # purple
        'Lupus': '#3498DB',     # blue
        'ChamI': '#95A5A6',     # gray
        'IC348': '#E67E22',     # orange (if exists)
        'USco': '#2ECC71',      # green
        'CrA': '#8B4513',       # brown
        'ChamII': '#E91E63',    # pink
    }
    
    # Process each region
    for region in regions:
        region_data = df[df['Region'] == region].copy()
        
        # Get dust masses
        mdust = pd.to_numeric(region_data[dust_mass_col], errors='coerce')
        mdust = mdust.dropna()
        mdust = mdust[mdust > 0]  # Only positive values
        
        if len(mdust) < 5:
            continue
        
        # Sort dust masses
        mdust_sorted = np.sort(mdust.values)
        
        # Calculate cumulative probability (survival function)
        # P(> M) = fraction of sources with dust mass > M
        n = len(mdust_sorted)
        cumulative_prob = 1 - np.arange(1, n + 1) / n
        
        # Get color
        color = colors.get(region, np.random.rand(3))
        
        # Calculate median and standard deviation for annotation
        median_mdust = np.median(mdust_sorted)
        std_mdust = np.std(mdust_sorted)
        
        # Plot the cumulative distribution
        ax.plot(mdust_sorted, cumulative_prob, linewidth=2.5, 
               color=color, label=f'{region}: {median_mdust:.0f}±{std_mdust:.0f}M⊕',
               alpha=0.8)
        
        # Add shaded region for uncertainty (using bootstrapping)
        # Simple approach: calculate 16th and 84th percentiles
        n_bootstrap = 100
        boot_probs = []
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(mdust_sorted, size=n, replace=True)
            boot_sorted = np.sort(boot_sample)
            boot_probs.append(1 - np.arange(1, n + 1) / n)
        
        # Calculate percentiles
        lower = np.percentile(boot_probs, 16, axis=0)
        upper = np.percentile(boot_probs, 84, axis=0)
        
        ax.fill_between(mdust_sorted, lower, upper, color=color, alpha=0.2)
        
        print(f"{region}: N={len(mdust)}, median={median_mdust:.1f} M⊕")
    
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel(r'$M_{\rm dust}$ [$M_\oplus$]', fontsize=16)
    ax.set_ylabel(r'$P(> M_{\rm dust})$', fontsize=16)
    ax.set_title('Cumulative Dust Mass Distribution by Star-Forming Region', fontsize=16)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.1, 200)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = paths.figures / 'cumulative_dust_mass_by_region.eps'
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"\nCumulative distribution figure saved to: {output_path}")
    
    output_path_png = paths.figures / 'cumulative_dust_mass_by_region.png'
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"Cumulative distribution figure saved to: {output_path_png}\n")
    
    plt.close()


def main():
    """Main analysis function."""
    
    # Read the original PPVII data (TSV format)
    print("Reading PP7-PPVII-original.tsv from PPVII website...")
    df = pd.read_csv(paths.data / 'PP7-PPVII-original.tsv', delimiter='\t')
    
    print(f"Total rows in full dataset: {len(df)}")
    
    # Use Standardized_Mdust_Mearth for dust mass (in Earth masses)
    # Use Mstar_PPVII for stellar mass (in solar masses)
    # The original file has correct decimal values (e.g., Sz112: Mstar=0.136, not 136)
    dust_mass_col = 'Standardized_Mdust_Mearth'
    stellar_mass_col = 'Mstar_PPVII'
    
    # 1. Analyze full sample
    perform_fit_and_plot(df, dust_mass_col, stellar_mass_col, 
                        '', 'Full Sample')
    
    # 2. Filter for young regions: Lupus, Taurus, and rOph
    # First, check what regions are available
    print("\nAvailable regions in dataset:")
    print(df['Region'].value_counts())
    
    # Filter for young regions (Lupus, Taurus, and variations of Ophiuchus)
    young_regions = df[df['Region'].isin(['Lupus', 'Taurus', 'rOph', 'Ophiuchus', 'ρOph'])].copy()
    
    print(f"\nYoung regions (Lupus, Taurus, rOph) total rows: {len(young_regions)}")
    
    # Analyze young sample with Ansdell et al. 2017 Lupus fit overplotted
    perform_fit_and_plot(young_regions, dust_mass_col, stellar_mass_col, 
                        '_young', 'Young Regions (Lupus, Taurus, rOph)', 
                        add_ansdell_fit=True)
    
    # 3. Create multipanel plot for all individual regions
    plot_all_regions(df, dust_mass_col, stellar_mass_col)
    
    # 4. Create cumulative dust mass distribution plot
    plot_cumulative_dust_mass(df, dust_mass_col)
    

if __name__ == '__main__':
    main()
