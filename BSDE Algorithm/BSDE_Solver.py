import tensorflow as tf
import time
from BSDE_functions import BSDE_functions


class BSDE_Solver:
    """
    A class that represents a collection of functions and their derivatives
    for a given system. The functions include f, g, b, sigma, H, and F, and
    their derivatives with respect to X or pi.

    Parameters
    ----------
    parameters : dict
        A dictionary containing the parameters (Q, R, S, A, B, C, D, G, L) used
        in the functions.

    Methods
    -------
    neural_network(isgamma=False) :
        Creates a neural network for pi or gamma.
    train(display_steps=False) :
        Trains the neural network model using the
        Deep BSDE method. Stores the control and bsde losses in the class.
    """

    def __init__(self, parameters):
        """
        Initialize the BSDE_Solver class with the given parameters.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the necessary parameters.
        """
        # Define the necessary functions from the class Functions
        self.functions = BSDE_functions(parameters)

        # Define the parameters
        self.Q = parameters['Q']
        self.R = parameters['R']
        self.S = parameters['S']
        self.A = parameters['A']
        self.B = parameters['B']
        self.C = parameters['C']
        self.D = parameters['D']
        self.G = parameters['G']
        self.L = parameters['L']

        # Initialise X, V, Z
        self.x_0 = parameters['x_0']
        self.V_0 = tf.Variable(tf.random.uniform((1, 1),
                                                 minval=-0.1,
                                                 maxval=0.1))
        self.Z_0 = tf.Variable(tf.random.uniform((2, 1),
                                                 minval=-0.1,
                                                 maxval=0.1))

        # Set batch size, number of iterations and N
        self.batch_size = parameters['batch_size']
        self.iteration_steps = parameters['iteration_steps']
        self.N = parameters['N']

        # Create N neural networks for pi and for gamma
        self.pi_networks = [
            self.neural_network(isgamma=False) for _ in range(self.N)
        ]
        self.gamma_networks = [
            self.neural_network(isgamma=True) for _ in range(self.N)
        ]

        # Set optimizers for each loss function
        self.optimizers_pi = [
            tf.keras.optimizers.Adam(learning_rate=1e-3) for _ in range(self.N)
        ]
        self.optimizer_gamma = tf.keras.optimizers.Adam(learning_rate=1e-2)

    def neural_network(self, hidden_layers=3, isgamma=False):
        """
        Create a neural network with uniform weight initialization.

        Parameters
        ----------
        isgamma : bool, optional, default: False
            If True, the output is gamma with size will be (n, 2, 2).
            If False the output is pi with size will be (n, 2, 1).

        Returns
        -------
        nn : tf.keras.Sequential
            The created neural network, which consists of 6 Dense layers with
            ReLU activation for the first three layers and uniform weight
            initialization.
        """

        # Specify the output sizes
        output_size = 2 if isgamma else 1

        # Create the neural network with uniform weight initialization
        nn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                12, activation='relu',
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.1, maxval=0.1),
                input_shape=(2, 1)),
        ])

        # Add `hidden_layers` - 1 hidden layers with ReLU activation
        for i in range(hidden_layers - 1):
            nn.add(tf.keras.layers.Dense(
                12, activation='relu',
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.1, maxval=0.1)))

        nn.add(tf.keras.layers.Dense(
            output_size,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.1, maxval=0.1)))
        nn.add(tf.keras.layers.Reshape((2, output_size)))

        return nn

    def train(self, display_steps=False):
        """
        Train the neural network model using the Deep BSDE method.
        Stores the control and bsde losses in the class.

        Parameters
        ----------
        display_steps : bool
            Whether to display progress. Default is False.

        Returns
        -------
            None.
        """

        # Initialise list for loss functions
        bsde_losses = []
        control_losses = []

        # Record the start time of the iteration step
        start_time = time.time()
        print("Training in progress")
        print("-"*50)

        # Iteration steps
        for iteration_step in range(1, self.iteration_steps + 1):

            # Modify learning rates at specified intervals
            if iteration_step % (self.iteration_steps // 3) == 0:

                self.optimizer_gamma.learning_rate.assign(
                    self.optimizer_gamma.learning_rate.numpy() / 10
                )
                for i in range(self.N):
                    self.optimizers_pi[i].learning_rate.assign(
                        self.optimizers_pi[i].learning_rate.numpy()
                    )

            # Optimising the Gamma Parameters & V_0, Z_0
            with tf.GradientTape() as tape:

                # Optimise V_0 and Z_0 as well
                tape.watch([self.V_0, self.Z_0])

                # Define training parameters
                trainable_param = [self.V_0, self.Z_0]
                for i in range(self.N):
                    trainable_param += self.gamma_networks[i].trainable_weights

                # Initialize X, V and Z
                X = tf.tile(
                    tf.expand_dims(self.x_0, 0), [self.batch_size, 1, 1])
                V = tf.tile(
                    tf.expand_dims(self.V_0, 0), [self.batch_size, 1, 1])
                Z = tf.tile(
                    tf.expand_dims(self.Z_0, 0), [self.batch_size, 1, 1])

                # Initialise Brownian motion
                dW = tf.math.sqrt(1 / self.N) * \
                    tf.random.normal(shape=(self.N, self.batch_size))

                # Perform Euler-Maruyama scheme
                for i in range(self.N):

                    # Compute dW_i
                    dW_i = tf.reshape(dW[i], (self.batch_size, 1, 1))

                    # Compute pi_i and gamma_i
                    pi = self.pi_networks[i](X)
                    gamma = self.gamma_networks[i](X)

                    # Update V_i
                    V = V + tf.matmul(
                        tf.transpose(Z, perm=[0, 2, 1]),
                        self.functions.sigma(X, pi)
                    ) * dW_i

                    # Update Z_i
                    q = tf.matmul(gamma, self.functions.sigma(X, pi))
                    Z = (Z - (1 / self.N) * self.functions.D_H(X, pi, Z, q)
                         + q * dW_i)

                    # Update X_i
                    X = (X + self.functions.b(X, pi) * (1 / self.N)
                         + self.functions.sigma(X, pi) * dW_i)

                # Compute the loss using the last values of X, V, Z
                loss_1 = tf.reduce_mean(
                    tf.square(V + self.functions.g(X)) +
                    0.5 * tf.reduce_sum(tf.square(Z + self.functions.D_g(X)),
                                        axis=1,
                                        keepdims=True)
                )

                # Store BSDE loss from this iteration step
                bsde_losses.append(loss_1.numpy())

            # Compute the gradients
            grads = tape.gradient(loss_1, trainable_param)

            # Use ADAM for optimisation
            self.optimizer_gamma.apply_gradients(zip(grads, trainable_param))

            # Optimising the Control Parameters
            with tf.GradientTape(persistent=True) as tape:

                # Initialize the loss for the current iteration
                loss_2 = 0

                # Initialise losses and gradients
                losses = [0 for _ in range(self.N)]
                gradients = [0 for _ in range(self.N)]

                # Define trainable parameters
                trainable_params = [0 for _ in range(self.N)]
                for i in range(self.N):
                    trainable_params[i] = self.pi_networks[i].trainable_weights

                # Initialize X
                X = tf.tile(
                    tf.expand_dims(self.x_0, 0), [self.batch_size, 1, 1])

                # Initialise Z
                Z = tf.tile(
                    tf.expand_dims(self.Z_0, 0), [self.batch_size, 1, 1])

                # Euler-Maruyama scheme and compute loss
                for i in range(self.N):

                    # Compute dW_i
                    dW_i = tf.reshape(dW[i], (self.batch_size, 1, 1))

                    # Compute pi_i and gamma_i
                    pi = self.pi_networks[i](X)
                    gamma = self.gamma_networks[i](X)

                    # Update Z_i
                    q = tf.matmul(gamma, self.functions.sigma(X, pi))
                    Z = (Z - (1 / self.N) * self.functions.D_H(X, pi, Z, q)
                         + q * dW_i)

                    # Update X_i
                    X = (X + self.functions.b(X, pi) * (1 / self.N)
                         + self.functions.sigma(X, pi) * dW_i)

                    # Compute loss_i
                    losses[i] += (
                        tf.reduce_mean(
                            tf.reduce_sum(
                                    tf.square(
                                        self.functions.D_F(X, pi, Z, gamma)
                                    ),
                                    axis=1,
                                    keepdims=True
                            )
                        )
                    )

            # Update parameters of the control network
            for i in range(self.N):

                # Define gradients
                gradients[i] = tape.gradient(losses[i], trainable_params[i])

                # Apply gradients
                self.optimizers_pi[i].apply_gradients(zip(gradients[i],
                                                          trainable_params[i]))

                # Compute the loss of the iteration as the sum of the losses
                loss_2 += losses[i]

            # Add the loss from this iteration to the list
            control_losses.append(loss_2)

            if display_steps:
                # Print the current iteration step and elapsed time
                if iteration_step % 100 == 0:

                    # Record the end time of the iteration step
                    end_time = time.time()

                    print(f"Iteration step {iteration_step}, "
                          f"time: {(end_time - start_time) / 60:.4f} minutes")

        # Store the bsde and control losses
        self.bsde_losses = bsde_losses
        self.control_losses = control_losses

        print("Training finished")
        return None

    def simulate_V(self):

        return None
