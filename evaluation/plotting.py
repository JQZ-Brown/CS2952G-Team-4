import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../fancyimpute")
from matrix_imputation import matrixFilter

# For test
lowres_sample = np.load("low_res_example.npy")
highres_samples = np.load("high_res_example.npy")

lowres_mat = lowres_sample[0].squeeze() # change this to a low-res matrix you want to visualize
highres_mat = highres_samples[0].squeeze() # change this to a high-res matrix
superres_mat = matrixFilter(lowres_sample)[0].squeeze() # change this to a super-res estimation

# normalization
lowres_mat /= np.max(lowres_mat)
highres_mat /= np.max(highres_mat)
superres_mat /= np.max(superres_mat)

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.title("Low-res $\mathbf{X}_{LR}$", fontsize = 15)
sbn.heatmap(lowres_mat, square = True, cbar = False, xticklabels = False, yticklabels = False)

plt.subplot(1, 3, 2)
plt.title("High-res $\mathbf{X}_{HR}$", fontsize = 15)
sbn.heatmap(highres_mat, square = True, cbar = False, xticklabels = False, yticklabels = False)

plt.subplot(1, 3, 3)
plt.title("Super-res $\mathbf{X}_{SR}$", fontsize = 15)
sbn.heatmap(superres_mat, square = True, cbar = False, xticklabels = False, yticklabels = False)
plt.show()