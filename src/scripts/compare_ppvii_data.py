#!/usr/bin/env python3
"""
Compare PP7-Manara data with our current disk fraction measurements
"""
import pandas as pd
import numpy as np

# Read PP7-Manara data
ppvii = pd.read_csv('../data/PP7-Manara.csv', sep=';')

# Count disks per region
print("=" * 80)
print("PPVII Chapter Data (Manara et al.) - Disk Statistics by Region")
print("=" * 80)

regions = ppvii['Region'].unique()
for region in regions:
    region_data = ppvii[ppvii['Region'] == region]
    
    # Count disk types
    # II = full disk, TD = transition disk, FS = full disk?, F = no disk, I = Class I (protostar)
    total = len(region_data)
    
    # Count sources with disks (II, TD, FS, or "b" for binary companion with disk)
    disk_sources = region_data[region_data['Disk'].isin(['II', 'TD', 'FS', 'II', 'Full', 'b'])].copy()
    n_disks = len(disk_sources)
    
    # Count diskless sources (F, III, or Debris/Ev. Trans.)
    no_disk = region_data[region_data['Disk'].isin(['F', 'III', 'Debris/Ev. Trans.'])].copy()
    n_no_disk = len(no_disk)
    
    # Exclude Class I (protostars)
    class_i = region_data[region_data['Disk'] == 'I']
    n_class_i = len(class_i)
    
    # Calculate disk fraction
    if (n_disks + n_no_disk) > 0:
        disk_frac = 100 * n_disks / (n_disks + n_no_disk)
    else:
        disk_frac = np.nan
    
    print(f"\n{region}:")
    print(f"  Total sources: {total}")
    print(f"  With disks (II/TD/FS): {n_disks}")
    print(f"  Without disks (F/III/Debris): {n_no_disk}")
    print(f"  Class I (excluded): {n_class_i}")
    if not np.isnan(disk_frac):
        print(f"  Disk fraction: {disk_frac:.1f}%")
    
    # Get age info if available
    ages = region_data['Age'].dropna().unique()
    if len(ages) > 0:
        print(f"  Typical age(s): {ages}")

print("\n" + "=" * 80)
print("Our Current Disk Fraction Data (Low-UV)")
print("=" * 80)

# Read our current data
low_uv = pd.read_csv('../data/disc_fraction_low_UV.csv', comment='#')
print("\n", low_uv.to_string(index=False))

print("\n" + "=" * 80)
print("Comparison - Overlapping Regions")
print("=" * 80)

# Map PPVII region names to our names
region_mapping = {
    'Lupus': ['Lupus', 'Lupus-on cloud', 'Lupus-off cloud'],
    'Taurus': ['Taurus'],
    'USco': ['Upp Sco'],
    'ChamI': ['Cham I'],
    'ChamII': ['Cham II'],
    'rOph': ['Ophiuchus'],
    'CrA': ['CrA']
}

print("\nRegions present in PPVII data:")
for ppvii_region in regions:
    # Find if this region is in our data
    matched = False
    for our_regions in region_mapping.values():
        if any(ppvii_region.lower() in r.lower() or r.lower() in ppvii_region.lower() 
               for r in our_regions):
            matched = True
            break
    
    marker = "✓" if matched else "✗"
    print(f"{marker} {ppvii_region}")

print("\n" + "=" * 80)
print("Key Observations:")
print("=" * 80)
print("""
1. The PPVII data contains individual stellar measurements, not aggregated 
   disk fractions for entire regions.

2. To properly compare with our data, we would need to:
   - Know the complete membership list for each region
   - Calculate disk fractions from the PPVII sample
   - Account for selection biases (e.g., ALMA-observed targets)

3. The PPVII chapter likely uses the same underlying data as:
   - Michel et al. 2021 (Gaia-based membership)
   - Ansdell et al. 2016-2018 (ALMA surveys)
   - Manara et al. 2016-2020 (accretion measurements)

4. Our current disk fraction data (Michel 2021, Pfalzner 2022) represents
   complete or nearly-complete surveys with Gaia-based membership, while
   the PPVII table focuses on sources with detailed disk/accretion measurements.
""")
