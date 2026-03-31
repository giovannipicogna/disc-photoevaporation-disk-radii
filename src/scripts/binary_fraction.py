import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

def calculate_effective_disc_fraction():
    """
    Calculates the effective initial protoplanetary disc fraction for a 
    stellar population (0.1 - 1.0 M_sun) based on Kroupa IMF and 
    binary separation distributions.
    """
    
    # --- 1. SETUP PARAMETERS ---
    
    # Mass Range (Solar Masses)
    M_MIN = 0.1
    M_BREAK = 0.5
    M_MAX = 1.0
    
    # Kroupa (2001) IMF Slopes (xi(m) ~ m^-alpha)
    ALPHA_LOW = 1.3
    ALPHA_HIGH = 2.3
    
    # Binary Fractions (observed in young regions)
    FBIN_M = 0.35  # M-dwarfs (0.1 - 0.5 M_sun)
    FBIN_G = 0.70  # Solar-type (0.5 - 1.0 M_sun)
    
    # Destructive Separation Range (AU)
    # Binaries in this range are assumed to destroy discs rapidly
    A_KILL_MIN = 3.0
    A_KILL_MAX = 30.0
    
    # Log-Normal Separation Parameters (log10(a/AU))
    # M-dwarfs (Winters et al. 2019): Peak ~20 AU
    MU_M = 1.30   
    SIGMA_M = 1.10
    
    # G-dwarfs (Raghavan et al. 2010): Peak ~50 AU
    MU_G = 1.70
    SIGMA_G = 1.68 

    # --- 2. CALCULATE POPULATION WEIGHTS (IMF) ---
    
    # Normalization constant calculation for IMF
    # C1 * 0.5^-1.3 = C2 * 0.5^-2.3  => C2 = C1 * 0.5^(2.3-1.3) = 0.5 * C1
    # We set C1 = 1.0 arbitrarily, then C2 = 0.5
    C1 = 1.0
    C2 = 0.5
    
    def imf(m):
        if m < M_BREAK:
            return C1 * m**(-ALPHA_LOW)
        else:
            return C2 * m**(-ALPHA_HIGH)

    # Integrate IMF to get relative number of stars in each bin
    n_m_dwarfs, _ = quad(imf, M_MIN, M_BREAK)
    n_g_dwarfs, _ = quad(imf, M_BREAK, M_MAX)
    
    total_stars = n_m_dwarfs + n_g_dwarfs
    f_num_m = n_m_dwarfs / total_stars
    f_num_g = n_g_dwarfs / total_stars
    
    print(f"--- Population Statistics ---")
    print(f"M-dwarfs ({M_MIN}-{M_BREAK} Msun): {f_num_m:.1%} of population")
    print(f"G-dwarfs ({M_BREAK}-{M_MAX} Msun): {f_num_g:.1%} of population")

    # --- 3. CALCULATE WEIGHTED BINARY FRACTION ---
    
    total_fbin = (f_num_m * FBIN_M) + (f_num_g * FBIN_G)
    print(f"Integrated Initial Binary Fraction: {total_fbin:.1%}")

    # --- 4. CALCULATE KILLER BINARY FRACTION ---
    
    # Function to get fraction of log-normal between min and max
    def get_fraction_in_range(mu, sigma, amin, amax):
        log_min = np.log10(amin)
        log_max = np.log10(amax)
        # CDF(max) - CDF(min) using standard normal distribution
        # We transform log10(a) to standard normal variable Z
        z_max = (log_max - mu) / sigma
        z_min = (log_min - mu) / sigma
        return norm.cdf(z_max) - norm.cdf(z_min)

    frac_kill_m = get_fraction_in_range(MU_M, SIGMA_M, A_KILL_MIN, A_KILL_MAX)
    frac_kill_g = get_fraction_in_range(MU_G, SIGMA_G, A_KILL_MIN, A_KILL_MAX)
    
    print(f"\n--- Destructive Range ({A_KILL_MIN}-{A_KILL_MAX} AU) ---")
    print(f"Fraction of M-binaries in killer zone: {frac_kill_m:.1%}")
    print(f"Fraction of G-binaries in killer zone: {frac_kill_g:.1%}")

    # Weighted fraction of ALL binaries that are 'killers'
    # We need to weight by the NUMBER of binaries in each bin, not just number of stars
    n_binaries_m = n_m_dwarfs * FBIN_M
    n_binaries_g = n_g_dwarfs * FBIN_G
    total_binaries = n_binaries_m + n_binaries_g
    
    weighted_killer_fraction = ((n_binaries_m * frac_kill_m) + 
                                (n_binaries_g * frac_kill_g)) / total_binaries
                                
    print(f"Weighted fraction of binaries that are destructive: {weighted_killer_fraction:.1%}")

    # --- 5. FINAL EFFECTIVE DISC FRACTION ---
    
    # P(Disc) = 1 - P(Killer System)
    # P(Killer System) = P(Binary) * P(Separation is Killer | Binary)
    
    p_killer_system = total_fbin * weighted_killer_fraction
    effective_disc_fraction = 1.0 - p_killer_system
    
    print(f"\n--- FINAL RESULT ---")
    print(f"Percentage of systems with discs destroyed: {p_killer_system:.1%}")
    print(f"EFFECTIVE INITIAL DISC FRACTION: {effective_disc_fraction:.1%}")

if __name__ == "__main__":
    calculate_effective_disc_fraction()
