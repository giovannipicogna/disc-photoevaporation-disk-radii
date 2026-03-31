"""
Quick diagnostic to verify legend color assignments in fig5.py
"""
import matplotlib.pyplot as plt
import numpy as np

# Define the exact same colors and labels as in fig5.py
datasets = [
    ("Compact discs", "#000000"),
    ("Compact discs (reduced PE)", "#009E73"),
    ("Viscously spreading discs", "#56B4E9"),
    ("Viscously spreading discs (reduced PE)", "#CC79A7"),
    ("Viscously spreading discs (no r_crit)", "#0072B2"),
    ("Low-UV (FUV<100 G$_0$)", "#D55E00"),
    ("High-UV (FUV>200 G$_0$)", "#E69F00"),
]

print("Expected legend order and colors:")
print("="*60)
for i, (label, color) in enumerate(datasets, 1):
    print(f"{i}. {label:50s} -> {color}")

# Create a simple plot to visualize
fig, ax = plt.subplots(figsize=(10, 6))

for i, (label, color) in enumerate(datasets):
    # Plot dummy data with the same marker style as fig5.py
    x = np.linspace(0, 20, 10) + i*0.3
    y = 80 - i*10 + np.random.randn(10)*2
    ax.plot(x, y, '.', color=color, markersize=8, label=label)

ax.legend(loc='best', ncol=2, frameon=True)
ax.set_xlabel('Age [Myr]')
ax.set_ylabel('Disk fraction [%]')
ax.set_title('Legend Color Verification')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/giovanni/Papers/disc-photoevaporation-disk-radii/src/scripts/legend_test.png', dpi=150)
print("\nVisualization saved to legend_test.png")

# Print RGB values for the blue colors to show they're distinct
print("\n" + "="*60)
print("Comparing the two blue colors:")
print("="*60)
print(f"Viscously spreading discs:           #56B4E9 = RGB(86, 180, 233) - sky blue")
print(f"Viscously spreading discs (no r_crit): #0072B2 = RGB(0, 114, 178) - darker navy blue")
print("\nThese are distinct colors from the Wong colorblind-safe palette.")
