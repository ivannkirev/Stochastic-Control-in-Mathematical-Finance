import tensorflow as tf
import matplotlib.pyplot as plt
from BSDE_Solver import BSDE_Solver

# Set parameters
parameters = {
    "Q": tf.constant([[3.0, 0.0], [0.0, 3.0]]),
    'R': tf.constant([[2.0, 0.0], [0.0, 2.0]]),
    'S': tf.constant([[0.0, 0.0], [0.0, 0.0]]),
    'A': tf.constant([[0.5, 0.3], [0.3, 0.5]]),
    'B': tf.constant([[0.2, 0.1], [0.1, 0.2]]),
    'C': tf.constant([[0.3, 0.1], [0.1, 0.3]]),
    'D': tf.constant([[0.1, 0.2], [0.2, 0.1]]),
    'G': tf.constant([[2.0, 1.0], [3.0, 2.0]]),
    'L': tf.constant([[0.3], [0.5]]),
    'N': 10,
    'batch_size': 50,
    'iteration_steps': 1500,
    'x_0': tf.Variable([[0.4], [0.5]])
}

# Define solver with the parameters
solver = BSDE_Solver(parameters)

# Train solver
solver.train(display_steps=True)

# Plot the loss functions
fig, axs = plt.subplots(1, 2, figsize=(20, 7.5))

# First plot
axs[0].plot(solver.bsde_losses)
axs[0].set_yscale('log')
axs[0].set_title("Loss Function of the BSDE")
axs[0].set_xlabel("Iteration step")
axs[0].set_ylabel("BSDE Loss")

# Second plot
axs[1].plot(solver.control_losses)
axs[1].set_yscale('log')
axs[1].set_title(r"Derivative of the Hamiltonian w.r.t $\pi$")
axs[1].set_xlabel("Iteration step")
axs[1].set_ylabel("Control Loss")

# Display
plt.show()
