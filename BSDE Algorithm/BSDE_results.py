import tensorflow as tf
from BSDE_Solver import BSDE_Solver

# Set parameters
parameters = {
    "Q": tf.constant([[0.0, 0.0], [0.0, 0.0]]),
    'R': tf.constant([[0.0, 0.0], [0.0, 0.0]]),
    'S': tf.constant([[0.0, 0.0], [0.0, 0.0]]),
    'A': tf.constant([[0.5, 0.3], [0.3, 0.5]]),
    'B': tf.constant([[0.5, 0.1], [0.1, 0.5]]),
    'C': tf.constant([[0.3, 0.1], [0.1, 0.3]]),
    'D': tf.constant([[0.6, 0.2], [0.2, 0.6]]),
    'G': -tf.constant([[2.0, 3.5], [3.5, 2.0]]),
    'L': -tf.constant([[0.3], [0.5]]),
    'N': 10,
    'batch_size': 1024,
    'iteration_steps': 1000,
    'x_0': tf.Variable([[1.8], [1.3]]),
    'lr_gamma': 1e-2,
    'lr_pi': 1e-3
}

# Define solver with the parameters
solver = BSDE_Solver(parameters)

# Train solver
solver.train(display_steps=True)

# Plot the loss functions
solver.plot("losses")

# Plot X V and Z for the primal problem
solver.plot("primal_results")
