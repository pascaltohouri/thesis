import tensorflow as tf
import math
from math import exp
import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal
from scipy import sparse
import matplotlib.pyplot as plt
import time
import threading

debug_mode = True
verbose_mode = False
################################################# Initialisation #######################################################

def reset_globals():
    global image_counter, reward_history, current_reward, N_input
    image_counter = 0
    reward_history = np.array([], dtype=[('image_counter', np.int64), ('current_reward', np.float64)])
    current_reward = 0
    N_input = 784
    print(f"### RESET_GLOBALS() - IMAGE {image_counter}:"
          f"\nGlobal variables have been reset.")

""" 
Define a function for loading and preparing the MNIST dataset

     Returns: (tuples) (x_train, y_train), (x_test, y_test)
"""
def load_mnist():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize the input images
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert labels to one-hot encoded vectors
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Print for readability
    print(f"\n### LOAD_MNIST() - IMAGE {image_counter}\n"
          f"MNIST data loaded successfully.\n"
          f"Training data shape: x: {x_train.shape}, y: {y_train.shape}\n"
          f"Test data shape: x: {x_test.shape}, y: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)

""" 
Define a function for distributing the N input or output points on a square surface in the YZ plane. 
The plane is translated along the x-axis by a distance specified by the 'x' parameter.

    Args:
    N (float): The number of points to distribute.
    side_length (float): The length of the square's sides.
    x (float): The x coordinate of the points along the x-axis.

    Return:
    points (np.array): A (N, 3) array of points on a square surface
"""


def distribute_points_on_square(N, side_length=1, x=0):
    # Calculate the number of rows and columns
    n = int(math.ceil(math.sqrt(N)))

    # Calculate the step size for even distribution
    step = side_length / (n - 1) if n > 1 else side_length / 2

    points = []
    for i in range(n):
        for j in range(n):
            y = i * step
            z = j * step
            points.append((x, y, z))
            if len(points) == N:
                return np.array(points)

    # Randomly remove excess points
    if len(points) > N:
        points = np.array(points)
        np.random.shuffle(points)
        return points[:N]

    return np.array(points)

""" 
Define a function for intialising the network at t=0 within the unit cube

    Args:
    x_min_vec (np.array): The starting vertex of the initialisation zone
    x_max_vec (np.array: The finishing vertex of the initialisation zone
    resolution (float): The number of points between the upper and lower bound of the grid in a single dimension
    (e.g. an x interval (np.linspace) from 0 to 1 with resolution 6 is [0, 0.2, 0.4, 0.6, 0.8, 1])

    Returns:
    f_coord (np.array): A (res, res, res, 4) array containing (x, y, z, f(.)) coords
"""

def pdf_initialisation(x_min_vec, x_max_vec, resolution=51):
    # Ensure x_min and x_max are 1D tensors with 3 elements
    x_min = tf.reshape(x_min_vec, [3])
    x_max = tf.reshape(x_max_vec, [3])

    # Create a grid over the specified domain for all dimensions
    x = np.linspace(x_min[0].numpy(), x_max[0].numpy(), resolution)
    y = np.linspace(x_min[1].numpy(), x_max[1].numpy(), resolution)
    z = np.linspace(x_min[2].numpy(), x_max[2].numpy(), resolution)

    # Create meshgrid (e.g. X is a (res, res, res) array containing only x coordinates of the unit cube mesh)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Create f_coord array (f_coord combines the separate meshes into a (res,res,res,4) array of (x,y,z,f(.)) entries)
    f_coord = np.zeros((resolution, resolution, resolution, 4))
    f_coord[:, :, :, 0] = X
    f_coord[:, :, :, 1] = Y
    f_coord[:, :, :, 2] = Z
    f_coord[:, :, :, 3] = 1

    print(f"\n### PDF_INITIALISATION() - Image {image_counter}:"
          f"\nf_coord {f_coord.shape} successfully initialised"
          f"\nf_coord first value: {f_coord[:1, :1, :1, :]}")
    return f_coord

""" 
Define a function for intialising the network at t=0 within the unit cube

    Args:
    lambda_param (float): The lambda parameter for the poisson distribution dictating the distribution's shape
    x_min_vec (np.array): The starting coordinate of the initialisation zone
    x_max_vec (np.array: The finishing coordinate of the initialisation zone

    Returns:
    G_medial (np.array): A (N, 3) array of points within a cube (x_1, y_i, z_i) for point i=1 to N (python i=0 to N-1)
"""

def graph_initialisation(lambda_param, x_min_vec, x_max_vec):
    # Ensure inputs are tensors with the correct shape
    x_min = tf.convert_to_tensor(x_min_vec, dtype=tf.float32)
    x_max = tf.convert_to_tensor(x_max_vec, dtype=tf.float32)
    lambda_param = tf.convert_to_tensor(lambda_param, dtype=tf.float32)

    # Ensure x_min and x_max are 1D tensors with 3 elements
    x_min = tf.reshape(x_min_vec, [3])
    x_max = tf.reshape(x_max_vec, [3])

    # Compute cuboid volume
    l = x_max - x_min
    volume = tf.reduce_prod(l)

    # Generate number of points using Poisson distribution
    N_medial = np.random.poisson(lam=lambda_param * volume.numpy())
    N_medial = tf.cast(N_medial, tf.int32)

    # Generate uniform points. x_min and x_max must be vectors for the tf.random.normal function
    G_medial = tf.random.uniform(shape=(N_medial, 3), minval=x_min_vec, maxval=x_max_vec)

    print(f"\n### GRAPH_INITIALISATION() - IMAGE {image_counter}:"
          f"\nG_medial {G_medial.shape} successfully initialised")
    return G_medial

"""
Define a function for concatenating the initial input, output and medial node coordinates

    Args:
    I_coord (np.array): Input coordinates array of shape (N_in,3)
    G_medial (np.array): Medial coordinates array of shape (N_medial,3)
    O_coords (np.array): Output coordinates array of shape (N_out,3)

    Returns:
    G (np.array): Concatenated array G of shape (N_total,3) where N_total = N_in + N_medial + N_out
    A: A (N,1) array of activation states; one state for each node
"""

def combine_fullgraph_and_states(I_coord, G_medial, O_coords):
    # concatenate the input, medial, and output arrays into a single (N_input+N_medial+N_output, 3) array
    G = np.concatenate((I_coord, G_medial, O_coords), axis=0)
    A = np.zeros((G.shape[0],1))

    print(f"\n### COMBINE_FULLGRAPH_AND_STATES() - IMAGE {image_counter}:"
         f"\nG shape: {G.shape} A shape {A.shape}")
    return G, A

################################################# Propagation ##########################################################

""" 
Define a function for flatting the subsequently indexed mnist image and updating the image counter by 1

    Args:
    A (np.array): The current state array (N, 1)
    x_train (6000, 28, 28): The array containing MNIST training images

    Returns:
    A (np.array): The new state array with the next mnist image taken as input (N, 1)
"""

def input_next_image(A, x_train):
    global reward_history
    global image_counter

    # Check if the index is valid
    if image_counter < 0 or image_counter >= len(x_train):
        raise ValueError(f"Image counter out of range. Check whether all images have been processed.")

    # Select the image
    selected_image = x_train[image_counter]

    # Flatten the image and reshape to (784, 1)
    flattened_image = selected_image.reshape((784, 1))

    # Increment the counter for the next call
    image_counter += 1

    # Replace the first N_input values in the A matrix
    A[:N_input] = flattened_image

    print(f"\n### INPUT_NEXT_IMAGE() - IMAGE {image_counter}:"
          f"\nImage {image_counter} successfully loaded"
          f"\nA's first 3 values:\n{A[:3,:]}"
          f"\nlast 3 values: \n{A[-3:,:]}"
          f"\nA shape: {A.shape}")
    return A

"""
Define a function for creating a weight matrix based on the coordinates in G

    Args:
    G (np.array): Point coordinates of shape (N, D), where N is the number of points
                      and D is the dimensionality of the space.
    mu (float): Scaling factor for the distance in the weight calculation.

    Returns:
    W (np.array): Weight matrix W of shape (N, N)
"""


def condition_weight_matrix(G, mu=1.0, decay_parameter=2):
    N, D = G.shape
    assert D == 3, "G should be a (N,3) array representing 3D points"
    W = sparse.lil_matrix((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            # Calculate the distance between each pair of 3D points
            d = np.sqrt(np.sum((G[i] - G[j]) ** 2))

            # Determine if an edge exists based on a random condition
            if np.random.random() < exp(-1*decay_parameter*d):
                weight = 1 / (1 + mu * d)       # Calculate the weight using the formula: w_ij = 1 / (1 + mu * d)
                W[i, j] = weight
                W[j, i] = weight        # Assuming undirected graph

    # Store the weight in the sparse weight matrix W (CSR format for efficient storage)
    W = W.tocsr()
    print(f"\n### CONDITION_WEIGHT_MATRIX() - IMAGE {image_counter}:"
          f"\nSparse W shape: {W.shape}")       # For sparse to dense conversion: W.toarray()
    return W


""""
Define a function for creating a bias vector based on the coordinates in G

    Args:
    G (np.array): Point coordinates of shape (N, D), where N is the number of points
                      and D is the dimensionality of the space.
    k (int): Number of nearest neighbors to consider. If None, all points are considered neighbors.

    Returns:
    b (np.array): Bias vector b of shape (N, 1)
"""

def condition_bias_vector(G, k=5):
    N, D = G.shape

    # Calculate pairwise Euclidean distances
    diff = G[:, np.newaxis, :] - G[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    if k is None:
        # If k is None, consider all points as neighbours (inefficient)
        b = np.mean(distances, axis=1)
    else:
        # If k is specified, consider only k nearest neighbours. Ensure k is less than N-1
        k = min(k, N - 1)
        nearest_distances = np.partition(distances, k, axis=1)[:, :k]
        b = 1 / np.mean(nearest_distances, axis=1)

    # Reshape b to (N, 1)
    b = b.reshape(-1, 1)

    print(f"\n### CONDITION_BIAS_VECTOR(): - IMAGE {image_counter}:"
          f"\nBias vector b shape: {b.shape}")
    return b

"""
Define a function to check the reward given a current state vector and the current y_train label

   Args:
    A (np.array): Initial state vector of shape (N, 1)
    y_train (60000, 1): Set of image labels corresponding to training data.

    Returns:
    current_reward (float): updated current reward.
"""
def check_reward(A, y_train):
    global image_counter, reward_history

    # Select the row corresponding to the current image_counter
    current_y = y_train[image_counter]
    correct_class = np.where(current_y == 1)[0][0]

    # A's final element is the last row and first column
    current_output = A[-1, 0]

    # The reward is the value of a transformed normal distribution centered on the correct classification value
    # E.g. A correct classification of 5 yields a +100 reward for a final state A[-1,0] of 5.
    def calculate_reward(current_output, correct_class, std_dev=10, max_reward=100):
        if np.isnan(current_output):
            raise ValueError("No valid network output to calculate reward.")
        else:
            return max_reward * exp((-1 / (2 * std_dev ** 2)) * (current_output - correct_class) ** 2)
    current_reward = calculate_reward(current_output, correct_class)

    # Append the current image counter and reward to the history of image counters and rewards
    new_row = np.array([(image_counter, current_reward)], dtype=reward_history.dtype)
    reward_history = np.append(reward_history, new_row)

    # Function to call the last 5 rewards for printing (or n<5 if fewer than 5 images are processed)
    def get_recent_reward_history(reward_history):
        # Ensure n is not larger than 5
        N = reward_history.shape[0]
        n = min(5, N)
        return reward_history[-n:]

    recent_reward = get_recent_reward_history(reward_history)

    print(f"\n### REWARD - IMAGE {image_counter}:\n"
          f"The final element in A: {current_output}\n"
          f"The correct classification: {correct_class}\n"  #f"The current image one-hot encoding vector: {current_y}\n"
          f"The current reward: {current_reward}\n"
          f"Last 5 rewards:\n "
          f"{recent_reward}\n"
          f"reward_history shape: {reward_history.shape}\n"
          f"###\n")

    return current_reward, reward_history

"""
Define a Propagator class to manage continual activation/state propagation in a separate thread to the update processes.

1. Propagator applies sigmoid activation thresholding to update node states (to +1 or -1),
2. These +1/-1 states propagate through the network using the weight matrix,
3. The class contains start, stop, and update methods, and retrieves current activation states.
4. Threading supports thread-safe operations a for concurrent execution with the update processes.
(Threading is handy because concurrent executions can disrupt the timing of "time_based_execution" (see main sec.))

Example error without threading:
1. The update function produces a new A vector
2. The propagation algorithm updates the old A
3. The old A (not the new A) is then obtained and a new image is input 
4. This is problematic as the old A vector relates to a completely different graph and thus has a different shape
(when compared with the new and improved A).

### OBSOLETE tanh activation function ###

    def propagate(self):
        # Main propagation loop running in the background
        while self.running:
            with self.condition:    # Thread-safe data access
                # print(f"W shape: {self.W.shape}, A shape: {self.A.shape}")
                # ^Add this line to check the propagator is working
                # Apply propagation operations
                while not self.updated and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                self.updated = False

                self.A = np.tanh(self.A + self.b)
                if sparse.issparse(self.W):
                    self.A = self.W.dot(self.A)
                else:
                    self.A = np.dot(self.W, self.A)
"""
class Propagator:
    # The constructor method receives initial values for A, b, and W
    def __init__(self, A, b, W):
        self.A = A  # The activation/state vector
        self.b = b  # The bias vector
        self.W = W  # The weight matrix
        self.running = True     # Flag that controls the propagation loop
        self.lock = threading.Lock()    # Lock the thread to avoid competing commands
        self.thread = None  # Holds the thread object
        self.condition = threading.Condition(self.lock)
        self.updated = False

    def propagate(self):
        # Main propagation loop running in the background
        while self.running:
            with self.condition:  # Ensure thread-safe data access
                # Apply propagation operations
                while not self.updated and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                self.updated = False

                # Sigmoid activation function
                sigmoid = lambda x: 1 / (1 + np.exp(-x))

                # print(f"Pre-sigmoid activations: {self.A}")
                # Apply sigmoid activation
                activated = sigmoid(self.A + self.b)

                # print(f"Post-sigmoid activations: {activated}")
                # Threshold the activated values
                self.A = np.where(activated >= 0.5, 1, -1)

                # print(f"Post-thresholding activations: {activated}")
                # Apply weight matrix
                if sparse.issparse(self.W):
                    self.A = self.W.dot(self.A)
                else:
                    self.A = np.dot(self.W, self.A)

                # Ensure A remains a column vector
                self.A = self.A.reshape(-1, 1)


    def start(self):
        # Start the propagation in a separate thread
        self.thread = threading.Thread(target=self.propagate)
        self.thread.start()

    def stop(self):
        # Stop propagating
        self.running = False    # Set flag to stop the propagation loop
        if self.thread:
            self.thread.join()  # Wait for the thread to finish

    def update(self, new_A, new_b, new_W):
        with self.condition:
            self.A = new_A.reshape(-1, 1)
            self.b = new_b.reshape(-1, 1)
            self.W = new_W
            self.updated = True
            self.condition.notify()

    def get_A(self):
        # Get a copy of the current state vector A
        with self.lock:    # Ensure thread-safe access
            return np.copy(self.A)  # return a copy to prevent external modifications

############################################## Update ##################################################################


""" 
Define a function for updating the PDF given a reward

    Args:
    G (np.array): Array of node positions, shape (N, 3)
    m (np.array): Array of node masses, shape (N,1)
    f_coord (callable): Current PDF with coordinates: (x,y,z,f(.)OLD)
    current_reward (float): Current reward

    Return:
    f_coord: Updated PDF with coordinates: (x,y,z,f(.)NEW)

"""

def update_PDF_scalar_field(f_coord, G, current_reward, m=None):
    if m is None:
        m = np.ones((G.shape[0], 1))

    def generate_summed_gaussians(f_coord, G, m):
        """
        Generate and sum Gaussian distributions centered at points in G.

        Parameters:
        - f_coord (np.array): array (res, res, res, 4) containing x, y, z, and f(x,y,z) values
        - G (np.array): array (N, 3) containing N 3D points (means for Gaussians)

        Returns:
        - summed_gaussian: 3D (res, res, res) array of the summed Gaussian distributions
        """

        # Extract x, y, z coordinates from f_coord. coords has shape (res, res, res, 3)
        coords = f_coord[:, :, :, :3]

        # Initialize summed_gaussian with zeros.
        # summed_gaussian has shape (res, res, res) and contains only '0' lots of times.
        summed_gaussian = np.zeros(f_coord.shape[:3])

        # Create as many Gaussians as there are points in G.
        # 'gaussian' is the (res, res, res) array of gaussian points centered at 'node'
        # summed_gaussian (res, res, res) keeps track of the cumulative/summed Gaussians
        for i, node in enumerate(G):
            gaussian = multivariate_normal.pdf(coords, mean=node, cov=np.eye(3) * 0.01)

            # m[i] doesn't differ between nodes but perhaps an interesting idea for the future?
            # Some neurons are more important than others. Reward-based weighting is a possibility.
            summed_gaussian += m[i] * gaussian
        return summed_gaussian

    summed_gaussian = generate_summed_gaussians(f_coord, G, m)  # Sum all Gaussians
    m_sum = np.sum(m)   # Compute normalisation factor
    if m_sum == 0:
        raise ValueError("m_sum is zero. This will cause division by zero in the Phi function.")

    def Phi(summed_gaussian, m_sum):    # Normalize the field
        return summed_gaussian / m_sum

    # Generate random scaling factor (alternate: alpha = np.random.normal(1, 2))
    alpha = 1
    f_current = f_coord[:, :, :, 3]

    # Update field.
    # When the current reward is large and positive, amplify the PDF around the node values (f+reward*phi)
    f_new = (f_current + alpha * current_reward * Phi(summed_gaussian, m_sum)) / (1 + alpha * current_reward)


    # Ensure the summed PDFs still integrate to 1 by dividing by their summed integrals (/1+alpha*current_reward)
    f_coord[:, :, :, 3] = f_new     # Replace the 4th value (index 3) with values from f_new

    #print(f"\n### UPDATE - IMAGE {image_counter}:\n"
    #      f"m's shape: {m.shape}, m's sum: {np.sum(m)}\n"
    #      f"G's shape: {G.shape} (N_total, 3)\n"
    #      f"summed_gaussian first: {summed_gaussian[:1, :1, :1]}, last: {summed_gaussian[-1:, -1:, -1:]}\n"
    #      f"f_current {f_current.shape}: {f_current[:1, :1, :1]}, last: {f_current[-1:, -1:, -1:]}\n"
    #      f"f_new first {f_new.shape}: {f_new[:1, :1, :1]}, last: {f_new[-1:, -1:, -1:]}\n"
    #      f"f_coord {f_coord.shape}, first: {f_coord[:1, :1, :1, :]}, last: {f_coord[-1:, -1:, -1:, :]}\n"
    #      f"###")

    return f_coord

############################################## Realisation #############################################################

""" 
Define a function for realising the graph from the saved f_coord PDF values

    Args:
    f_coord (np.array): Array of (x,y,z,f(.)) entries, shape (51, 51, 51, 4))
    I_coords (np.array): Input coordinate array of shape (N_in,3)
    O_coords (np.array): Output coordinate array of shape (N_out,3)
    max_rows (float): The maximum number of realised nodes (rows in G)

    Return:
    G (np.array): A new set of nodes (N_in+N_medial+N_out, 3)

"""


def realisation(f_coord, I_coord, O_coord, max_rows=200, sampling_radius=3):
    # Extract the scalar field values (4th dimension).
    # 'f_coord' is a (res, res, res, 4) array containing res^3 (x,y,z,f(.)) values
    f_current = f_coord[:, :, :, 3]

    # Optional: add random noise so the amplified values don't dominate in subsequent realisation stages.
    # noise = np.random.uniform(-0.0000;pl, 0.0000, size=f_current.shape)  # Small uniform noise
    f_current = f_current #+ noise

    # Find local maxima
    local_max = ndimage.maximum_filter(f_current, size=3)
    mask = (f_current == local_max)

    # Exclude border elements
    mask[0, :, :] = mask[-1, :, :] = False
    mask[:, 0, :] = mask[:, -1, :] = False
    mask[:, :, 0] = mask[:, :, -1] = False

    maxima_indices = np.argwhere(mask)

    # Function to sample a point near a maximum based on local PDF values
    def sample_near_maximum(index, radius):
        x, y, z = index
        x_min, x_max = max(0, x - radius), min(f_current.shape[0], x + radius + 1)
        y_min, y_max = max(0, y - radius), min(f_current.shape[1], y + radius + 1)
        z_min, z_max = max(0, z - radius), min(f_current.shape[2], z + radius + 1)

        local_region = f_current[x_min:x_max, y_min:y_max, z_min:z_max]
        probs = local_region.flatten()
        probs /= probs.sum()  # Normalize to get probabilities

        indices = np.array(list(np.ndindex(local_region.shape)))
        chosen_index = indices[np.random.choice(len(indices), p=probs)]

        return np.array([x_min, y_min, z_min]) + chosen_index

    # Sample points near maxima
    G_medial = np.array([sample_near_maximum(idx, sampling_radius) for idx in maxima_indices])

    # Normalize G_medial coordinates to [0, 1] range
    G_medial = G_medial / (np.array(f_coord.shape[:3]) - 1)

    # Limit the number of medial points if necessary
    if len(G_medial) > max_rows:
        G_medial = G_medial[np.random.choice(len(G_medial), max_rows, replace=False)]

    # Combine with input and output coordinates
    G_new, A = combine_fullgraph_and_states(I_coord, G_medial, O_coord)

    print(f"\n### REALISATION:\n"
          f"f_coord's shape: {f_coord.shape}\n"
          f"New G_medial's shape: {G_medial.shape}\n"
          f"I's shape: {I_coord.shape}\n"
          f"O's shape: {O_coord.shape}\n"
          f"New G's shape: {G_new.shape}")

    return G_new, A


############################################## Visualisation ###########################################################

""" 
Define a function for displaying the graph (viewed down from above the XY plane). 
Also displays an XY 'slice' of the PDF (currently the slice at Z=0.5).

    Args:
    G (np.array): Array of node positions, shape (N, 3)
    f_coord (callable): Current PDF with coordinates: (x,y,z,f(.))

    Return:
    None: This function displays a plot but does not return any value.

"""
def show_pdf_and_nodes(f_coord, G):
    f_current = f_coord[:, :, :, 3]
    domain_range = [0, 1]
    resolution = 51
    interval = np.linspace(domain_range[0], domain_range[1], resolution)

    # Create a grid over the specified domain for all dimensions at once.
    # np.linspace creates a bounded {domain_range} interval with {resolution} evenly spaced coordinates
    # The tuple "X, Y, Z" becomes a set of 3 meshes of shape (resolution, resolution, resolution) x3.
    # The three coordinate meshes are separate objects
    X, Y, Z = np.meshgrid(interval, interval, interval)

    # Visualize a slice of the PDF result (XY plane at Z=0.5)
    # middle slice along Z axis
    plt.figure(figsize=(10, 8))
    middle_z = Z.shape[2] // 2
    plt.contourf(X[:, :, middle_z], Y[:, :, middle_z], f_current[:, :, middle_z], levels=100, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title(f'Slice of the Current PDF (Z=0.5) - IMAGE {image_counter}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(G[:, 0], G[:, 1], color='red', s=50, label='Node coordinates')
    plt.legend()
    plt.show()

    print(f"\nPlot {image_counter} complete")

################################################  Main  ################################################################
# Time-based model execution
def time_based_execution(x_train, y_train, I_coord, O_coord, f_coord, G, A, b, W, duration=60000, input_interval=5):
    # Record the start time
    start_time = time.time()

    # Start propagation with the intialised activation states.
    propagator = Propagator(A, b, W)
    propagator.start()

    try:
        while time.time() - start_time < duration:
            # Record the runtime and the most recent whole second
            t = time.time() - start_time
            current_whole_second = int(t)

            # Every 'input_interval' seconds...
            if current_whole_second % input_interval == 0:

                # Check that all the objects have the correct shape
                #print(f"b shape: {b.shape}, W shape: {W.shape}, A shape: {A.shape}, G shape: {G.shape}")

                # Retrieve the current activation state vector from the background propagator
                A = propagator.get_A()
                # print(f"\n### PROPAGATION - IMAGE {image_counter} (runtime = {t:.3f}):\n"
                #      f"A's first and last values:\n"
                #      f" {A[:3, :]}\n "
                #      f"  ...\n "
                #      f"{A[-3:, :]}\n"
                #      f"###")

                # Check the reward by comparing the correct classification against A's final value
                # print(f"\nruntime = {t:.3f}")
                current_reward, reward_history = check_reward(A, y_train)

                # Use the reward to update the spatial PDF and realise a new 3D graph
                f_coord = update_PDF_scalar_field(f_coord, G, current_reward, m=None)
                G, A = realisation(f_coord, I_coord, O_coord)
                show_pdf_and_nodes(f_coord, G)

                # Condition the weights and biases using the new graph
                # print(f"\nruntime = {t:.3f}")
                W = condition_weight_matrix(G, mu=0.5)
                b = condition_bias_vector(G, k=5)

                # Input the next image into the new, empty activation/state vector
                # print(f"\nruntime = {t:.3f}")
                A = input_next_image(A, x_train)

                # Update the background propagator with new A, b, and W
                propagator.update(A, b, W)

            time.sleep(0.01)
    finally:
        propagator.stop()

""""
Define a function for implementing the hitherto sub-functions
"""

def main():
    # Reset global variables, initialise input and output arrays.
    reset_globals()  # Set the image counter to 0, load the data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    I_coord = distribute_points_on_square(N=N_input, side_length=1, x=0)  # Initialise the node coordinates
    O_coord = np.array([1, 0.5, 0.5])
    O_coord = O_coord.reshape(1, 3)


    # Initialise the PDF within a unit cube
    x_min_vec = [0, 0, 0]  # Initialise the uniform PDF
    x_max_vec = [1, 1, 1]
    f_coord = pdf_initialisation(x_min_vec, x_max_vec, resolution=51)

    # Initialise the graph from the PDF and display both
    G_medial = graph_initialisation(lambda_param=100, x_min_vec=x_min_vec, x_max_vec=x_max_vec)

    # Initialise the medial node coordinates and combine with input-output nodes
    G, A = combine_fullgraph_and_states(I_coord, G_medial, O_coord)
    print(f"\n### INITIALISATION - IMAGE {image_counter}:\n"
          f"I shape: {I_coord.shape}\n"
          f"O shape: {O_coord.shape}\n"
          f"G_medial shape:{G_medial.shape}\n"
          f"Total number of nodes (I+O+Medial): {I_coord.shape[0] + O_coord.shape[0]  + G_medial.shape[0] }")
    show_pdf_and_nodes(f_coord, G)

    # Condition weights and biases on node positions
    W = condition_weight_matrix(G, mu=0.5)
    b = condition_bias_vector(G, k=5)

    # Repeat the above steps under timed conditions:
#   print(f"b shape: {b.shape}, W shape: {W.shape}, A shape: {A.shape}, G shape: {G.shape}")

    # It's showtime!
    time_based_execution(x_train, y_train, I_coord, O_coord, f_coord, G, A, b, W, duration=60000)
    show_pdf_and_nodes(f_coord, G)

if __name__ == "__main__":
    main()
