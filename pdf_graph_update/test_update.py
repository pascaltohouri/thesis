"""
print(f"Always print {var1}" +
      f"{' optional debug info ' + str(var2) if debug_mode else ''}" +
      f"{' verbose info ' + str(var3) if verbose_mode else ''}" +
      f" Always print this too {var4}")
"""

r"""
To activate tensorflow and tensroflow_probability:
1. open command prompt
2. type: C:\Users\pasca\Anaconda3\Scripts\activate tf
3. Now the tf environment is active, type: python -c "import tensorflow as tf; print(tf.__version__)"
4. Install tensorflow_probability: pip install tensorflow-probability
5. After installation, verify it: python -c "import tensorflow_probability as tfp; print(tfp.__version__)"

Slicing: array[start:stop:step],
Omitting parameters:
    array[:] selects all elements
    array[start:] goes from start to the end
    array[:stop] goes from the beginning to stop
    array[::step] uses the specified step over the entire array

Examples:
    array[1:4] selects elements at indices 1, 2, and 3
    array[-3:] selects the last three elements
    array[::2] selects every other element
    array[::-1] reverses the array
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
debug_mode = True
verbose_mode = True

def update_PDF_scalar_field(f_coord, G, current_reward, m=None):
    if m is None:
        m = np.ones((G.shape[0], 1))
    # Number of points in G: N_total = G.shape[0]
    print("m shape:", m.shape)
    print("m sum:", np.sum(m))

    def generate_summed_gaussians(f_coord, G, m):
        """
        Generate summed Gaussian distributions centered at points in G.

        Parameters:
        - f_coord: numpy array of shape (res, res, res, 4) containing x, y, z, and f(x,y,z) values
        - G: numpy array of shape (N, 3) containing N 3D points (means for Gaussians)

        Returns:
        - summed_gaussian: 3D (res, res, res) array of the summed Gaussian distributions
        """

        print(f"The node coordinate array G has shape: {G.shape}")
        print(f"The f_coord array has shape: {f_coord.shape}")

        # Extract x, y, z coordinates from f_coord
        # coords has shape (res, res, res, 3)
        coords = f_coord[:, :, :, :3]

        # Initialize summed_gaussian with zeros
        # summed_gaussian has shape (res, res, res) and contains only '0' lots of times
        summed_gaussian = np.zeros(f_coord.shape[:3])

        # Create as many Gaussians as there are points in G
        # 'gaussian is the (res, res, res) array of gaussian points centered at 'node'
        # summed_gaussian (res, res, res) keeps track of the cumulative Gaussians
        for i, node in enumerate(G):
            gaussian = multivariate_normal.pdf(coords, mean=node, cov=np.eye(3) * 0.01)
            summed_gaussian += m[i] * gaussian # m[i] doesn't differ between nodes but perhaps an interesting idea for the future? Some neurons are more equal than others!

        return summed_gaussian

    # Sum all Gaussians
    summed_gaussian = generate_summed_gaussians(f_coord, G, m)

    # Compute normalization factor
    m_sum = np.sum(m)
    if m_sum == 0:
        raise ValueError("m_sum is zero. This will cause division by zero in the Phi function.")

    # Normalize the field
    def Phi(summed_gaussian, m_sum):
        return summed_gaussian / m_sum

    # Generate random scaling factor
    alpha = np.random.normal(0.5, 1)

    # Update field
    # When the current reward is large and positive, amplify the PDF around the node values (f+reward*phi)
    # Add random noise so the amplified values don't dominate in subsequent realisation stages (f+reward*alpha*phi)
    # Ensure the summed PDFs still integrate to one by dividing by their summed integrals (/1+alpha*current_reward)
    f_current = f_coord[:,:,:,3]
    f_new = (f_current + alpha * current_reward * Phi(summed_gaussian, m_sum)) / (1 + alpha * current_reward)

    # Replace the 4th value (index 3) with values from f_new
    f_coord[:, :, :, 3] = f_new

    return summed_gaussian, f_new, f_coord




# Example usage:
domain_range = [0, 1]
resolution = 51

# Create a grid over the specified domain for all dimensions at once
# np.linspace creates a bounded {domain_range} interval with {resolution} evenly spaced coordinates
interval = np.linspace(domain_range[0], domain_range[1], resolution)

# The tuple "X, Y, Z" becomes a set of 3 meshes of shape (resolution, resolution, resolution) x3
# The three coordinate meshes are separate objects
X, Y, Z = np.meshgrid(interval, interval, interval)

np.random.seed(0)
G = np.random.uniform(0, 1, (100, 3))
f_current = np.random.uniform(0, 1, (51, 51, 51))

# Visualize a slice of the Uniform distribution result (XY plane at Z=0.5)
plt.figure(figsize=(10, 8))
middle_z = Z.shape[2] // 2  # middle slice along Z axis
plt.contourf(X[:, :, middle_z], Y[:, :, middle_z], f_current[:, :, middle_z], levels=100, cmap='viridis')
plt.colorbar(label='Density')
plt.title('Slice of the Uniform Distribution (Z=0.5)')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(G[:, 0], G[:, 1], color='red', s=50, label='Distribution centers')
plt.legend()
plt.show()


# Utilize numpy magic so the 3 meshes combine and each sub array (row) is a 3D point
# The final mesh_coords array has a (resolution^3, 3) shape; evenly spaced 3D coords from the combined meshes
f_coord = np.stack([X, Y, Z, f_current], axis=-1)

m = np.random.uniform(0.5, 1.5, (100, 1))  # Create a non-zero m
summed_gaussian, f_new, f_coord = update_PDF_scalar_field(f_coord, G, current_reward=0.5, m=m)

# Visualize a slice of the summed Gaussian result (XY plane at Z=0.5)
plt.figure(figsize=(10, 8))
middle_z = Z.shape[2] // 2  # middle slice along Z axis
plt.contourf(X[:, :, middle_z], Y[:, :, middle_z], summed_gaussian[:, :, middle_z], levels=100, cmap='viridis')
plt.colorbar(label='Density')
plt.title('Slice of Summed Gaussians (Z=0.5)')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(G[:, 0], G[:, 1], color='red', s=50, label='Node coordinates')
plt.legend()
plt.show()

# Visualize a slice of the updated f function (XY plane at Z=0.5)
plt.figure(figsize=(10, 8))
plt.contourf(X[:, :, middle_z], Y[:, :, middle_z], f_new[:, :, middle_z], levels=1000, cmap='viridis')
plt.colorbar(label='Density')
plt.title('Slice of Updated f Function (Z=0.5)')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(G[:, 0], G[:, 1], color='red', s=50, label='Distribution centers')
plt.legend()
plt.show()
