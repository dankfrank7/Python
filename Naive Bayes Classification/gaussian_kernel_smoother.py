import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generating some sample data
np.random.seed(0)
x = np.linspace(-3, 3, 100)
y = np.sin(x) + np.random.normal(0, 0.3, size=x.shape)

# Gaussian Kernel Smoothing function
def gaussian_kernel_smoothing(x, y, bandwidth):
    smooth_y = np.zeros_like(y)
    for i in range(len(x)):
        weights = norm.pdf(x, x[i], bandwidth)
        smooth_y[i] = np.sum(weights * y) / np.sum(weights)
    return smooth_y

# Apply smoothing
smoothed_y = gaussian_kernel_smoothing(x, y, bandwidth=0.25**2)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data', color='blue', alpha=0.6)
plt.plot(x, smoothed_y, label='Smoothed Data', color='red')
plt.title('Gaussian Kernel Smoothing')
plt.legend()
plt.show()