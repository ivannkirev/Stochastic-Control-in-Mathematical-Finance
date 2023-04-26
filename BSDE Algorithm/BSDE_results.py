import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from BSDE_Solver import BSDE_Solver
from Runge_Kutta_Solver import Runge_Kutta_Solver
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove warnings

n = 20
# Set parameters
parameters = {
    "Q": tf.constant([[0.1, 0.2], [0.2, 0.1]]),
    'R': tf.constant([[0.1, 0.2], [0.2, 0.1]]),
    'S': tf.constant([[0.0, 0.0], [0.0, 0.0]]),
    'A': tf.constant([[0.0, 0.4], [0.4, 0.0]]),
    'B': tf.constant([[0.3, 0.0], [0.0, 0.3]]),
    'C': tf.constant([[0.3, 0.2], [0.2, 0.3]]),
    'D': tf.constant([[0.2, 1.0], [1.0, 0.2]]),
    'G': -tf.constant([[0.6, 0.4], [0.4, 0.6]]),
    'L': -tf.constant([[0.3], [0.1]]),
    'N': n,
    'batch_size': 1024,
    'iteration_steps': 2000,
    'x_0': tf.Variable([[1.8], [1.3]]),
    'lr_gamma': 1e-1,
    'lr_pi': 1e-2
}

parameters_RK = {
    "Q": np.array([[0.1, 0.2], [0.2, 0.1]]),
    'R': np.array([[0.1, 0.2], [0.2, 0.1]]),
    'S': np.array([[0.0, 0.0], [0.0, 0.0]]),
    'A': np.array([[0.0, 0.4], [0.4, 0.0]]),
    'B': np.array([[0.3, 0.0], [0.0, 0.3]]),
    'C': np.array([[0.3, 0.2], [0.2, 0.3]]),
    'D': np.array([[0.2, 0.1], [0.1, 0.2]]),
    'G': np.array([[0.6, 0.4], [0.4, 0.6]]),
    'L': np.array([[0.3], [0.1]]),
    'N': n,
}

# Define solver with the parameters
bsde_solver = BSDE_Solver(parameters)
ode_solver = Runge_Kutta_Solver(parameters_RK)

bsde_solver.train(display_steps=True)
bsde_solver.plot("losses")
# bsde_solver.plot("primal_results")

X, V_bsde, Z = bsde_solver.simulate()

V_ode = ode_solver.compute_V(X)

plt.figure(figsize=(15, 7.5))
plt.plot(V_ode, label="ODE")
plt.plot(V_bsde, label="BSDE")
plt.title("V")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(15, 7.5))
plt.plot(abs(V_ode + V_bsde))
plt.title("Error")
plt.grid()
plt.show()
