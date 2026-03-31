import numpy as np
import scipy.interpolate as interpolate
from scipy.stats import norm

def calculate_population_statistics():
    """
    Generates a stellar population to calculate Initial Multiplicity Fraction
    and Effective Initial Disc Fraction with error propagation.
    """
    # --- 1. SETUP PARAMETERS ---
    n_runs = 10000  # Number of stars per realization
    n_iterations = 1000  # Number of MC realizations for error estimation
    
    # Mass Grid for PDF (same as provided script)
    masses_grid = np.linspace(0.1, 1.0, n_runs)
    
    # Kroupa (2001) IMF Setup
    IMF_Kroupa = []
    for m in masses_grid:
        if m < 0.5:
            IMF_Kroupa.append([m, m**(-1.3)])
        else:
            IMF_Kroupa.append([m, m**(-2.3)])
    
    IMF_Kroupa = np.array(IMF_Kroupa)
    weights = IMF_Kroupa[:, 1]
    prob_dist = weights / np.sum(weights)

    # Multiplicity Fractions (Mean, Std Dev)
    # M-dwarfs (0.1 - 0.5 Msun) - User provided
    mf_m = (0.26, 0.03) 
    # K-dwarfs (0.5 - 0.7 Msun) - Duchene & Kraus (2013)
    mf_k = (0.34, 0.04)
    # G-dwarfs (0.7 - 1.0 Msun) - User provided
    mf_g = (0.41, 0.03)

    # Separation Distribution Parameters (log10 a/AU)
    # M-dwarfs (Winters+2019): Peak ~20 AU
    mu_m, sig_m = 1.30, 1.10
    # Solar-type (Raghavan+2010): Peak ~50 AU
    mu_sol, sig_sol = 1.70, 1.68

    # Destructive Range (AU)
    # Binaries in this range destroy discs rapidly
    kill_min, kill_max = 3.0, 30.0

    # --- 2. MONTE CARLO SIMULATION ---
    results_imf_frac = []
    results_disc_frac = []

    np.random.seed(1234) # Reproducibility

    print(f"Running {n_iterations} iterations of {n_runs} stars each...")

    for _ in range(n_iterations):
        # A. Generate Masses
        masses = np.random.choice(masses_grid, size=n_runs, p=prob_dist)

        # B. Sample Multiplicity Rates (Systematic Error)
        # We sample the 'global' rate for this universe from the error distribution
        # to propagate the uncertainty in the measured rates.
        rate_m = np.clip(np.random.normal(*mf_m), 0, 1)
        rate_k = np.clip(np.random.normal(*mf_k), 0, 1)
        rate_g = np.clip(np.random.normal(*mf_g), 0, 1)

        # C. Assign Binary Status
        # Create probability array based on mass
        prob_binary = np.zeros(n_runs)
        mask_m = (masses < 0.5)
        mask_k = (masses >= 0.5) & (masses < 0.7)
        mask_g = (masses >= 0.7)

        prob_binary[mask_m] = rate_m
        prob_binary[mask_k] = rate_k
        prob_binary[mask_g] = rate_g

        is_binary = np.random.random(n_runs) < prob_binary
        
        # D. Assign Separations
        separations = np.zeros(n_runs) # 0 for singles
        
        # M-dwarf binaries
        n_m_bin = np.sum(is_binary & mask_m)
        if n_m_bin > 0:
            separations[is_binary & mask_m] = 10**np.random.normal(mu_m, sig_m, n_m_bin)
            
        # Solar-type binaries (K+G)
        n_sol_bin = np.sum(is_binary & (mask_k | mask_g))
        if n_sol_bin > 0:
            separations[is_binary & (mask_k | mask_g)] = 10**np.random.normal(mu_sol, sig_sol, n_sol_bin)

        # E. Calculate Statistics
        
        # Initial Multiplicity Fraction
        imf_fraction = np.sum(is_binary) / n_runs
        results_imf_frac.append(imf_fraction)

        # Effective Disc Fraction
        # Disc is destroyed if binary separation is in destructive range
        is_killer = is_binary & (separations > kill_min) & (separations < kill_max)
        n_survivors = n_runs - np.sum(is_killer)
        
        disc_fraction = n_survivors / n_runs
        results_disc_frac.append(disc_fraction)

    # --- 3. OUTPUT ---
    mean_imf = np.mean(results_imf_frac)
    std_imf = np.std(results_imf_frac)
    
    mean_disc = np.mean(results_disc_frac)
    std_disc = np.std(results_disc_frac)

    print("\n" + "="*40)
    print(f"POPULATION STATISTICS (0.1 - 1.0 Msun)")
    print("="*40)
    print(f"Initial Multiplicity Fraction: {mean_imf:.1%} +/- {std_imf:.1%}")
    print(f"Effective Initial Disc Fraction: {mean_disc:.1%} +/- {std_disc:.1%}")
    print(f"(Assumed destructive range: {kill_min} - {kill_max} AU)")

if __name__ == "__main__":
    calculate_population_statistics()
