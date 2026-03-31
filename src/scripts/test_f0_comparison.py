#!/usr/bin/env python3
"""
Compare different f0 approaches for disk fraction fitting
"""
import numpy as np
from scipy.optimize import curve_fit

# Read LOW-UV data (skip header, load columns: age, age_lower, age_upper, frac, frac_lower, frac_upper)
data_low = np.loadtxt('../data/disc_fraction_low_UV.csv', delimiter=',', skiprows=1,
                      usecols=(1, 4), dtype=float)  # age (col 1) and disc fraction (col 4)
age_low = data_low[:, 0]
frac_low = data_low[:, 1]

# Read HIGH-UV data  
data_high = np.loadtxt('../data/disc_fraction_high_UV.csv', delimiter=',', skiprows=1,
                       usecols=(1, 4), dtype=float)  # age (col 1) and disc fraction (col 4)
age_high = data_high[:, 0]
frac_high = data_high[:, 1]

# Define fitting functions
def exp_decay_fixed_f0(t, tau, f0=87.6):
    """Fixed f0 (1-parameter fit)"""
    return f0 * np.exp(-t / tau)

def exp_decay_free(t, f0, tau):
    """Free f0 (2-parameter fit)"""
    return f0 * np.exp(-t / tau)

print("="*80)
print("COMPARISON: Fixed f0 vs Free f0")
print("="*80)

# ----------------------------------------------------------------------------
# CURRENT APPROACH: Fixed f0 = 87.6% for both environments
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("CURRENT (Fixed f0 = 87.6% for both)")
print("="*80)

# LOW-UV
sort_idx_low = np.argsort(age_low)
age_sorted_low = age_low[sort_idx_low]
frac_sorted_low = frac_low[sort_idx_low]

popt_low_fixed, pcov_low_fixed = curve_fit(
    exp_decay_fixed_f0, age_sorted_low, frac_sorted_low,
    p0=[np.mean(age_low)], maxfev=2000
)
tau_low_fixed = popt_low_fixed[0]
tau_low_fixed_err = np.sqrt(np.diag(pcov_low_fixed))[0]

print(f"\nLOW-UV:")
print(f"  f0 = 87.6% (FIXED)")
print(f"  τ  = {tau_low_fixed:.2f} ± {tau_low_fixed_err:.2f} Myr")

# HIGH-UV
sort_idx_high = np.argsort(age_high)
age_sorted_high = age_high[sort_idx_high]
frac_sorted_high = frac_high[sort_idx_high]

popt_high_fixed, pcov_high_fixed = curve_fit(
    exp_decay_fixed_f0, age_sorted_high, frac_sorted_high,
    p0=[np.mean(age_high)], maxfev=2000
)
tau_high_fixed = popt_high_fixed[0]
tau_high_fixed_err = np.sqrt(np.diag(pcov_high_fixed))[0]

print(f"\nHIGH-UV:")
print(f"  f0 = 87.6% (FIXED)")
print(f"  τ  = {tau_high_fixed:.2f} ± {tau_high_fixed_err:.2f} Myr")

print(f"\nRatio: High-UV is {tau_low_fixed/tau_high_fixed:.2f}× faster dispersal")

# ----------------------------------------------------------------------------
# OPTION 1: Environment-specific fixed f0 (Polnitzky/Pfalzner values)
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("OPTION 1 (Environment-specific fixed f0: Low-UV=77%, High-UV=65%)")
print("="*80)

# LOW-UV with f0=77%
popt_low_opt1, pcov_low_opt1 = curve_fit(
    lambda t, tau: exp_decay_fixed_f0(t, tau, f0=77.0),
    age_sorted_low, frac_sorted_low,
    p0=[np.mean(age_low)], maxfev=2000
)
tau_low_opt1 = popt_low_opt1[0]
tau_low_opt1_err = np.sqrt(np.diag(pcov_low_opt1))[0]

print(f"\nLOW-UV:")
print(f"  f0 = 77.0% (FIXED - Polnitzky 2024 Sco-Cen)")
print(f"  τ  = {tau_low_opt1:.2f} ± {tau_low_opt1_err:.2f} Myr")
print(f"  Change from 87.6%: Δτ = {tau_low_opt1 - tau_low_fixed:+.2f} Myr ({100*(tau_low_opt1/tau_low_fixed - 1):+.1f}%)")

# HIGH-UV with f0=65%
popt_high_opt1, pcov_high_opt1 = curve_fit(
    lambda t, tau: exp_decay_fixed_f0(t, tau, f0=65.0),
    age_sorted_high, frac_sorted_high,
    p0=[np.mean(age_high)], maxfev=2000
)
tau_high_opt1 = popt_high_opt1[0]
tau_high_opt1_err = np.sqrt(np.diag(pcov_high_opt1))[0]

print(f"\nHIGH-UV:")
print(f"  f0 = 65.0% (FIXED - Pfalzner 2024 best fit)")
print(f"  τ  = {tau_high_opt1:.2f} ± {tau_high_opt1_err:.2f} Myr")
print(f"  Change from 87.6%: Δτ = {tau_high_opt1 - tau_high_fixed:+.2f} Myr ({100*(tau_high_opt1/tau_high_fixed - 1):+.1f}%)")

print(f"\nRatio: High-UV is {tau_low_opt1/tau_high_opt1:.2f}× faster dispersal")

# ----------------------------------------------------------------------------
# OPTION 2: Free f0 - let data decide
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("OPTION 2 (Free f0 - data determines both f0 and τ)")
print("="*80)

# LOW-UV
popt_low_free, pcov_low_free = curve_fit(
    exp_decay_free, age_sorted_low, frac_sorted_low,
    p0=[87.6, np.mean(age_low)],  # Initial guess
    bounds=([0, 0], [100, 50]),    # f0 in [0,100]%, tau in [0,50] Myr
    maxfev=5000
)
f0_low_free = popt_low_free[0]
tau_low_free = popt_low_free[1]
errors_low_free = np.sqrt(np.diag(pcov_low_free))
f0_low_free_err = errors_low_free[0]
tau_low_free_err = errors_low_free[1]

print(f"\nLOW-UV:")
print(f"  f0 = {f0_low_free:.1f} ± {f0_low_free_err:.1f}% (FREE - fitted)")
print(f"  τ  = {tau_low_free:.2f} ± {tau_low_free_err:.2f} Myr")
print(f"  Change from fixed 87.6%:")
print(f"    Δf0 = {f0_low_free - 87.6:+.1f}%")
print(f"    Δτ  = {tau_low_free - tau_low_fixed:+.2f} Myr ({100*(tau_low_free/tau_low_fixed - 1):+.1f}%)")

# HIGH-UV
popt_high_free, pcov_high_free = curve_fit(
    exp_decay_free, age_sorted_high, frac_sorted_high,
    p0=[87.6, np.mean(age_high)],  # Initial guess
    bounds=([0, 0], [100, 50]),     # f0 in [0,100]%, tau in [0,50] Myr
    maxfev=5000
)
f0_high_free = popt_high_free[0]
tau_high_free = popt_high_free[1]
errors_high_free = np.sqrt(np.diag(pcov_high_free))
f0_high_free_err = errors_high_free[0]
tau_high_free_err = errors_high_free[1]

print(f"\nHIGH-UV:")
print(f"  f0 = {f0_high_free:.1f} ± {f0_high_free_err:.1f}% (FREE - fitted)")
print(f"  τ  = {tau_high_free:.2f} ± {tau_high_free_err:.2f} Myr")
print(f"  Change from fixed 87.6%:")
print(f"    Δf0 = {f0_high_free - 87.6:+.1f}%")
print(f"    Δτ  = {tau_high_free - tau_high_fixed:+.2f} Myr ({100*(tau_high_free/tau_high_fixed - 1):+.1f}%)")

print(f"\nRatio: High-UV is {tau_low_free/tau_high_free:.2f}× faster dispersal")

# ----------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print("\n                                  LOW-UV                  |  HIGH-UV")
print("                         f0 (%)      τ (Myr)        |  f0 (%)      τ (Myr)")
print("-" * 80)
print(f"CURRENT (87.6% both)     87.6 (fixed) {tau_low_fixed:5.2f} ± {tau_low_fixed_err:.2f}  |  87.6 (fixed) {tau_high_fixed:5.2f} ± {tau_high_fixed_err:.2f}")
print(f"Option 1 (env-specific)  77.0 (fixed) {tau_low_opt1:5.2f} ± {tau_low_opt1_err:.2f}  |  65.0 (fixed) {tau_high_opt1:5.2f} ± {tau_high_opt1_err:.2f}")
print(f"Option 2 (free f0)       {f0_low_free:4.1f} ± {f0_low_free_err:3.1f}   {tau_low_free:5.2f} ± {tau_low_free_err:.2f}  |  {f0_high_free:4.1f} ± {f0_high_free_err:3.1f}   {tau_high_free:5.2f} ± {tau_high_free_err:.2f}")
print("=" * 80)

print("\nKEY INSIGHTS:")
print("-" * 80)
if f0_low_free < 87.6:
    print(f"• LOW-UV data prefers f0 = {f0_low_free:.1f}%, LOWER than theoretical 87.6%")
else:
    print(f"• LOW-UV data prefers f0 = {f0_low_free:.1f}%, close to theoretical 87.6%")

if f0_high_free < f0_low_free:
    print(f"• HIGH-UV f0 = {f0_high_free:.1f}% is {f0_low_free - f0_high_free:.1f}% lower than low-UV")
    print("  → Consistent with early UV destruction reducing initial observable fraction")
else:
    print(f"• HIGH-UV and LOW-UV have similar f0 (~{f0_high_free:.1f}%)")

print(f"• When f0 is free, τ changes by {100*(tau_low_free/tau_low_fixed - 1):+.1f}% (low-UV) and {100*(tau_high_free/tau_high_fixed - 1):+.1f}% (high-UV)")
print(f"• Correlation: fixing f0 higher → forces faster dispersal (lower τ) to match data")
print("=" * 80)
