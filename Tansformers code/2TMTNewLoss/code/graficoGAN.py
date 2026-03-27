import numpy as np
import matplotlib.pyplot as plt

# Data
C = np.array([
    1.00e-17, 3.00e-17, 5.00e-17, 7.00e-17, 9.00e-17,
    1.00e-16, 3.00e-16, 5.00e-16, 7.00e-16, 9.00e-16,
    1.00e-15, 3.00e-15, 5.00e-15, 7.00e-15, 9.00e-15,
    1.00e-14, 3.00e-14, 5.00e-14, 7.00e-14, 9.00e-14
])

blurred = np.array([
    0.6973573673, 0.6963061005, 0.6946729574, 0.6936752658, 0.6918861336,
    0.6913142118, 0.6823407055, 0.6711840138, 0.6632017472, 0.6554987687,
    0.6507885532, 0.5847179768, 0.5429390615, 0.5126582507, 0.4900203646,
    0.4821667853, 0.3950791848, 0.3612707381, 0.3408792891, 0.3270594770
])

reconstructed = np.array([
    0.9278777344, 0.9200608249, 0.9121638690, 0.9053007058, 0.8996420936,
    0.8971051937, 0.8573914191, 0.8258301976, 0.8015750734, 0.7800085850,
    0.7696509528, 0.6570751842, 0.6018786798, 0.5656797403, 0.5405209040,
    0.5314277750, 0.4406603759, 0.4059458582, 0.3850089426, 0.3705270080
])

# Sort in DESCENDING order (largest C first)
order = np.argsort(C)[::-1]
C = C[order]
blurred = blurred[order]
reconstructed = reconstructed[order]

# X positions
x = np.arange(len(C))

# Labels formatted as scientific notation
xlabels = [f"{c:.0e}" for c in C]

# Plot
plt.figure(figsize=(8, 4.5))
width = 0.4

plt.bar(x - width/2, reconstructed, width=width, label="Reconstructed")
plt.bar(x + width/2, blurred, width=width, label="Turbulence")

plt.ylabel("SSIM")
plt.xlabel(r"$C_n^2$")
plt.xticks(x, xlabels, rotation=90)
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.show()
