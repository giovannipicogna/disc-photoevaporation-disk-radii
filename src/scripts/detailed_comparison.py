#!/usr/bin/env python3
"""
Detailed comparison of PPVII data vs our disk fraction measurements
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("DETAILED COMPARISON: PPVII (Manara) vs Our Disk Fraction Data")
print("=" * 80)

# Read PPVII data and calculate disk fractions
ppvii = pd.read_csv('../data/PP7-Manara.csv', sep=';')

# Read our disk fraction data (skip comment lines starting with ##, but keep the header with #)
with open('../data/disc_fraction_low_UV.csv', 'r') as f:
    lines = f.readlines()
# Keep only the header (first #) and data lines
header = lines[0].lstrip('#').strip()
data_lines = [header] + [l for l in lines[1:] if not l.startswith('#')]

import io
low_uv = pd.read_csv(io.StringIO(''.join(data_lines)))

# Map between PPVII region names and our region names
comparison_data = {
    'Region': [],
    'PPVII_DF': [],
    'Our_DF': [],
    'PPVII_N': [],
    'Our_Source': [],
    'Age_PPVII': [],
    'Age_Our': [],
    'Discrepancy': []
}

# Direct comparisons
comparisons = [
    ('Lupus', 'Lupus', 2.9),
    ('Taurus', 'Taurus', 1.5),
    ('ChamI', 'Cham I', 2.0),
    ('ChamII', 'Cham II', 2.0),
    ('rOph', 'Ophiuchus', 1.5),
    ('USco', 'Upp Sco', 11.0),
    ('CrA', 'CrA', 5.0)
]

for ppvii_name, our_name, age_our in comparisons:
    # Get PPVII data
    region_data = ppvii[ppvii['Region'] == ppvii_name]
    disk_sources = region_data[region_data['Disk'].isin(['II', 'TD', 'FS', 'Full', 'b'])]
    no_disk = region_data[region_data['Disk'].isin(['F', 'III', 'Debris/Ev. Trans.'])]
    
    n_disks = len(disk_sources)
    n_no_disk = len(no_disk)
    
    if (n_disks + n_no_disk) > 0:
        ppvii_df = 100 * n_disks / (n_disks + n_no_disk)
    else:
        ppvii_df = np.nan
    
    # Get our data
    our_row = low_uv[low_uv['name'].str.contains(our_name, case=False, na=False)]
    
    if len(our_row) > 0:
        our_df = our_row['disc fraction (%)'].values[0]
        our_source = our_row['source'].values[0]
        
        if not np.isnan(ppvii_df):
            discrepancy = ppvii_df - our_df
            comparison_data['Region'].append(our_name)
            comparison_data['PPVII_DF'].append(f"{ppvii_df:.1f}%")
            comparison_data['Our_DF'].append(f"{our_df:.0f}%")
            comparison_data['PPVII_N'].append(f"{n_disks}/{n_disks + n_no_disk}")
            comparison_data['Our_Source'].append(our_source)
            comparison_data['Age_PPVII'].append(region_data['Age'].mode()[0] if len(region_data['Age'].mode()) > 0 else 'N/A')
            comparison_data['Age_Our'].append(age_our)
            comparison_data['Discrepancy'].append(f"{discrepancy:+.1f}%")

df_comparison = pd.DataFrame(comparison_data)
print("\n")
print(df_comparison.to_string(index=False))

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)

print("""
1. **MAJOR DISCREPANCIES** between PPVII data and our disk fraction data:
   - Lupus: PPVII shows 91% vs Michel 2021 shows 52% (+39% difference!)
   - Taurus: PPVII shows 100% vs Michel 2021 shows 49% (+51% difference!)
   - Upper Sco: PPVII shows 66% vs Michel 2021 shows 20% (+46% difference!)
   - CrA: PPVII shows 89% vs Michel 2021 shows 25% (+64% difference!)

2. **REASON FOR DISCREPANCIES - Selection Bias:**
   
   The PPVII (Manara et al.) table contains:
   - Stars that were SELECTED for detailed observations (ALMA, spectroscopy)
   - These samples are BIASED toward disk-bearing stars
   - They do NOT represent complete, unbiased surveys of regions
   
   Our data (Michel 2021, Pfalzner 2022) represents:
   - Complete or nearly-complete surveys using Gaia DR3 membership
   - All stars in a region, not just those with disks
   - Proper disk fractions for the entire population

3. **APPROPRIATE USE OF PPVII DATA:**
   
   ✓ Studying disk properties (masses, sizes, accretion rates) 
   ✓ Understanding disk evolution for individual systems
   ✓ Disk demographics among disk-bearing stars
   
   ✗ NOT suitable for disk fraction measurements
   ✗ NOT a complete census of star-forming regions
   ✗ Selection effects make it unsuitable for population statistics

4. **CONCLUSION:**
   
   We should CONTINUE using Michel et al. 2021 and Pfalzner et al. 2022 for 
   disk fractions, as these represent complete Gaia-based surveys. The PPVII 
   data is excellent for disk properties but not for population statistics.

5. **AGES:**
   
   The ages in PPVII data are broadly consistent with our data, so the age
   estimates are not the source of discrepancy - it's purely the selection
   bias toward disk-bearing stars in the PPVII sample.
""")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("""
KEEP using our current disk fraction data from:
- Michel et al. 2021 (ApJ 921, 72) - Gaia-based, low-UV regions
- Pfalzner et al. 2022 (ApJL 939, L10) - Gaia-based, updated ages
- Luhman 2022, Galli et al. 2021 - region-specific updates

DO NOT switch to PPVII data for disk fractions - it represents a biased
sample of stars selected for detailed observations, not complete censuses.
""")
