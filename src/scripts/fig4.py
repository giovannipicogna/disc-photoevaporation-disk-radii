"""
Publication-ready script for surface density evolution comparison.

This script creates a two-panel figure comparing disc surface density evolution
between different photoevaporation prescriptions, showing sigma vs radius at
various percentages of disc lifetime.

Author: Giovanni Picogna
Date: October 2025 - Optimized for publication
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import paths
import subprocess
import matplotlib.font_manager as font_manager

# Try to use science plots style if available
try:
    import scienceplots
    plt.style.use('science')
    print("Using scienceplots style for publication quality")
except ImportError:
    print("scienceplots not available, using default matplotlib style")

# Try seaborn for color palettes
try:
    import seaborn as sns
    sns.set_palette("viridis")
except ImportError:
    print("seaborn not available, using matplotlib defaults")

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================
AU = 1.496e13  # cm to AU conversion
OUTPUT_TIMESTEP_YR = 10000  # Each output represents 10,000 years

try:
    kpse_cp = subprocess.run(['kpsewhich', '-var-value', 'TEXMFDIST'], capture_output=True, check=True)
    font_loc1 = os.path.join(kpse_cp.stdout.decode('utf8').strip(), 'fonts', 'opentype', 'public', 'tex-gyre')
    print(f'loading TeX Gyre fonts from "{font_loc1}"')
    font_dirs = [font_loc1]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    plt.rcParams['font.family'] = 'TeX Gyre Termes'
except (FileNotFoundError, subprocess.CalledProcessError):
    print("kpsewhich not found; falling back to default serif font")
    plt.rcParams['font.family'] = 'serif'
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
    'figure.figsize': [20./3., 5.],  # MNRAS single column width
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# ============================================================================
# DATA READING FUNCTIONS
# ============================================================================


def read_grid_file(grid_path):
    """Read the grid.dat file and return radius in AU."""
    try:
        grid_data = np.loadtxt(grid_path)
        radius_cm = grid_data  # Assuming grid.dat contains radius in cm
        radius_au = radius_cm / AU
        return radius_au
    except Exception as e:
        print(f"Error reading grid file: {e}")
        return None

def read_sigma_file(sigma_path):
    """Read a sigma*.dat file and return surface density."""
    try:
        sigma_data = np.loadtxt(sigma_path)
        return sigma_data
    except Exception as e:
        print(f"Error reading {sigma_path}: {e}")
        return None


def get_sigma_files(run_path):
    """Get all sigma*.dat files and sort them numerically."""
    sigma_pattern = os.path.join(run_path, "sigma*.dat")
    sigma_files = glob.glob(sigma_pattern)
    
    # Extract numbers from filenames for proper sorting
    def extract_number(filename):
        base = os.path.basename(filename)
        # Remove 'sigma' and '.dat' to get the number
        number_str = base.replace('sigma', '').replace('.dat', '')
        try:
            return int(number_str)
        except ValueError:
            return 0
    
    sigma_files.sort(key=extract_number)
    return sigma_files


def read_critical_radius(run_path):
    """Read the critical radius (r1) from parameters.dat file."""
    params_path = os.path.join(run_path, "parameters.dat")
    
    if not os.path.exists(params_path):
        print(f"Warning: parameters.dat not found in {run_path}")
        return None
    
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        if 'r1' in params:
            r1_au = float(params['r1'])
            print(f"Found critical radius r1 = {r1_au:.1f} AU")
            return r1_au
        else:
            print("Warning: r1 parameter not found in parameters.dat")
            return None
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in parameters.dat: {e}")
        return None
    except Exception as e:
        print(f"Error reading parameters.dat: {e}")
        return None


def main():
    """Main function to create surface density evolution comparison figure."""
    # ========================================================================
    # DATA PATHS AND CONFIGURATION
    # ========================================================================
    # Column definitions: key -> (path template suffix, column title)
    prescriptions = [
        ('new',           paths.data / "single_internal_{au}_factor10",
         r'$r_1$ dependent'),
        ('old',           paths.data / "single_internal_{au}_factor10_norcrit",
         'unconstrained'),
        ('ext_G01',       paths.data / "single_internal_{au}_factor10_external_G01",
         r'$G_0 = 1$'),
        ('ext_G010',      paths.data / "single_internal_{au}_factor10_external_G010",
         r'$G_0 = 10$'),
        ('ext_G0100',     paths.data / "single_internal_{au}_factor10_external_G0100",
         r'$G_0 = 100$'),
    ]

    # Row definitions: au tag and label
    row_configs = [
        ('5au',  '5 AU'),
        ('20au', '20 AU'),
        ('80au', '80 AU'),
    ]

    n_rows = len(row_configs)
    n_cols = len(prescriptions)

    # ========================================================================
    # FIGURE CREATION AND FORMATTING
    # ========================================================================
    # MNRAS double-column width: 6.97 inches; height scaled for 3 rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.97, 5.),
                             sharex=True, sharey='row')
    fig.subplots_adjust(left=0.08, bottom=0.10, right=0.92, top=0.84,
                        wspace=0.10, hspace=0.28)

    # Define percentages for all plots
    percentages = np.array([0, 15, 30, 45, 60, 75, 90, 92, 94, 96, 100])

    # Color map for different times (using modern matplotlib API)
    try:
        colors = plt.colormaps['plasma'](np.linspace(0, 1, len(percentages)))
    except (KeyError, AttributeError):
        import matplotlib.cm as cm
        colors = cm.get_cmap('plasma')(np.linspace(0, 1, len(percentages)))

    # Subplot label sequence: a) … o)
    subplot_labels = [f'{chr(ord("a") + i)})' for i in range(n_rows * n_cols)]

    # ========================================================================
    # PROCESS ALL CONFIGURATIONS
    # ========================================================================
    for row_idx, (au_tag, row_label) in enumerate(row_configs):
        print(f"\n=== Processing {row_label} configuration ===")

        for col_idx, (pres_key, path_template, _) in enumerate(prescriptions):
            run_path = str(path_template).replace('{au}', au_tag)
            ax = axes[row_idx, col_idx]

            print(f"Analyzing {pres_key} run: {run_path}")

            # Check if directory exists
            if not os.path.exists(run_path):
                print(f"Error: Directory {run_path} does not exist!")
                ax.text(0.5, 0.5, "Data not\nfound",
                        transform=ax.transAxes, ha='center', va='center',
                        style='italic', fontsize=SMALL_SIZE)
                continue

            # Read grid file
            grid_path = os.path.join(run_path, "grid.dat")
            radius_au = read_grid_file(grid_path)

            if radius_au is None:
                print(f"Failed to read grid file for {pres_key}!")
                continue

            print(f"Read grid with {len(radius_au)} radial points")
            print(f"Radius range: {radius_au.min():.1f} - {radius_au.max():.1f} AU")

            # Read critical radius from parameters.dat
            critical_radius_au = read_critical_radius(run_path)

            # Get all sigma files
            sigma_files = get_sigma_files(run_path)

            if not sigma_files:
                print(f"No sigma*.dat files found for {pres_key}!")
                continue

            print(f"Found {len(sigma_files)} sigma files")

            # Determine disk lifetime (number of outputs)
            total_outputs = len(sigma_files) - 1

            # Calculate disc lifetime in Myr
            disc_lifetime_myr = (total_outputs * OUTPUT_TIMESTEP_YR) / 1e6
            print(f"{pres_key} disc lifetime: {disc_lifetime_myr:.2f} Myr")

            # Set subplot title: column name (top row only) + lifetime
            if row_idx == 0:
                ax.set_title(f'{prescriptions[col_idx][2]}\n({disc_lifetime_myr:.2f} Myr)',
                             fontsize=SMALL_SIZE, pad=3)
            else:
                ax.set_title(f'({disc_lifetime_myr:.2f} Myr)',
                             fontsize=SMALL_SIZE, pad=3)

            # Select files at key percentages of disk lifetime
            selected_indices = []
            for percent in percentages:
                index = int((percent / 100.0) * (total_outputs - 1))
                selected_indices.append(index)

            print(f"Selected output indices: {selected_indices}")

            # Plot surface density evolution
            for i, (percent, index) in enumerate(zip(percentages, selected_indices)):
                sigma_file = sigma_files[index]
                sigma = read_sigma_file(sigma_file)

                if sigma is None:
                    print(f"Failed to read {sigma_file}")
                    continue

                if len(sigma) != len(radius_au):
                    print(f"Warning: sigma file {sigma_file} has {len(sigma)} points, "
                          f"but grid has {len(radius_au)} points")
                    min_len = min(len(sigma), len(radius_au))
                    sigma = sigma[:min_len]
                    r_plot = radius_au[:min_len]
                else:
                    r_plot = radius_au

                # Plot only non-zero values (in log scale)
                mask = sigma > 0
                if np.any(mask):
                    # Only add percentage label in the very first subplot
                    label = f'{percent}%' if (row_idx == 0 and col_idx == 0) else None
                    ax.loglog(r_plot[mask], sigma[mask],
                              color=colors[i], linewidth=1.0, label=label)
                    print(f"Plotted {percent}% of lifetime: {np.sum(mask)} points")
                else:
                    print(f"Warning: No positive sigma values for {percent}% of lifetime")

            # Add critical radius vertical line if available
            if critical_radius_au is not None:
                critical_label = 'Critical radius' if (row_idx == 0 and col_idx == 0) else None
                ax.axvline(critical_radius_au, color='gray', linestyle='--',
                           linewidth=1.0, alpha=0.8, label=critical_label)
                print(f"Added critical radius line at {critical_radius_au:.1f} AU")

            # Axis labels: bottom row gets x-label, left column gets y-label
            if row_idx == n_rows - 1:
                ax.set_xlabel('Radius [AU]')
            else:
                ax.tick_params(axis='x', labelbottom=False)

            if col_idx == 0:
                ax.set_ylabel(r'$\Sigma$ [g/cm$^2$]')
            else:
                ax.tick_params(axis='y', labelleft=False)

            # Row label on the right side of the last column
            if col_idx == n_cols - 1:
                ax.text(1.05, 0.5, row_label,
                        transform=ax.transAxes, fontweight='bold',
                        va='center', ha='left', rotation=90, fontsize=SMALL_SIZE)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.01, 1000)
            ax.set_ylim(1e-4, 1e6)

            # Subplot label
            label_idx = row_idx * n_cols + col_idx
            ax.text(0.95, 0.95, subplot_labels[label_idx], transform=ax.transAxes,
                    fontweight='bold', va='top', ha='right', fontsize=SMALL_SIZE)

    # ========================================================================
    # SHARED LEGEND
    # ========================================================================
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.99), loc='upper center',
               fontsize=SMALL_SIZE, frameon=True, fancybox=True, shadow=False,
               ncol=6)

    # ========================================================================
    # FIGURE FINALIZATION
    # ========================================================================
    output_file = paths.figures / 'Fig4.png'
    fig.savefig(output_file, format='png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved as: {output_file}")
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    print(f"Figure dimensions: {fig_w:.1f}\" x {fig_h:.1f}\"")
    print("Optimized for double-column A4 scientific publication")
    print(f"Grid layout: {n_rows} rows x {n_cols} columns")


if __name__ == "__main__":
    main()
