import numpy as np
from scipy.stats import norm

def calculate_uncertainties(n_iterations=10000):
    """
    Propagates errors from IMF slopes to the effective disc fraction
    using Monte Carlo sampling.
    """
    # --- Fixed Parameters ---
    M_MIN, M_BREAK, M_MAX = 0.1, 0.5, 1.0
    FBIN_M, FBIN_G = 0.35, 0.70
    
    # Destructive Range (10-100 AU)
    A_MIN, A_MAX = 3.0, 30.0
    
    # Separation Distributions (Log-Normal parameters)
    # M-dwarfs: Peak ~20 AU
    MU_M, SIGMA_M = 1.30, 1.10
    # G-dwarfs: Peak ~50 AU
    MU_G, SIGMA_G = 1.70, 1.68
    
    # --- Pre-calculate Killer Probabilities (Fixed) ---
    def get_p_kill(mu, sigma):
        z_max = (np.log10(A_MAX) - mu) / sigma
        z_min = (np.log10(A_MIN) - mu) / sigma
        return norm.cdf(z_max) - norm.cdf(z_min)

    P_KILL_M = get_p_kill(MU_M, SIGMA_M) # Prob. M-binary is destructive
    P_KILL_G = get_p_kill(MU_G, SIGMA_G) # Prob. G-binary is destructive
    
    # --- Monte Carlo Simulation ---
    np.random.seed(42)
    
    # Sample IMF slopes from Normal distributions
    alpha_low_samples = np.random.normal(1.3, 0.5, n_iterations)
    alpha_high_samples = np.random.normal(2.3, 0.5, n_iterations)
    
    disc_fractions = []
    
    for a_low, a_high in zip(alpha_low_samples, alpha_high_samples):
        
        # 1. Integrate IMF to get Number counts (N_m, N_g)
        # Ensure continuity: C2 = C1 * 0.5^(a_high - a_low)
        # N ~ (m^(1-alpha)) / (1-alpha)
        
        # M-dwarf Number (0.1 - 0.5)
        # (Assuming C1=1)
        term_m = (0.5**(1-a_low) - 0.1**(1-a_low)) / (1-a_low)
        
        # G-dwarf Number (0.5 - 1.0)
        # Scale by continuity constant C2
        C2 = 0.5**(a_high - a_low)
        term_g = C2 * (1.0**(1-a_high) - 0.5**(1-a_high)) / (1-a_high)
        
        # 2. Calculate Population Fractions
        total_stars = term_m + term_g
        f_m = term_m / total_stars
        f_g = term_g / total_stars
        
        # 3. Calculate Global Binary Fraction for this realization
        f_bin_total = (f_m * FBIN_M) + (f_g * FBIN_G)
        
        # 4. Calculate Weighted Killer Fraction
        # We need the fraction of ALL binaries that are killers
        n_bin_m = f_m * FBIN_M
        n_bin_g = f_g * FBIN_G
        
        # Weighted average of P_KILL based on where the binaries are
        p_kill_weighted = (n_bin_m * P_KILL_M + n_bin_g * P_KILL_G) / (n_bin_m + n_bin_g)
        
        # 5. Effective Disc Fraction
        # P(Disc Lost) = P(Binary) * P(Killer | Binary)
        p_disc_lost = f_bin_total * p_kill_weighted
        disc_fractions.append(1.0 - p_disc_lost)
        
    # --- Results ---
    mean_val = np.mean(disc_fractions)
    std_val = np.std(disc_fractions)
    
    print(f"Effective Initial Disc Fraction: {mean_val:.1%} +/- {std_val:.1%}")

if __name__ == "__main__":
    calculate_uncertainties()
