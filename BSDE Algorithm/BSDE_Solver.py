import tensorflow as tf
import time
from BSDE_functions import BSDE_functions
import numpy as np


class BSDE_Solver:
    """
    This class provides a solver for the BSDE using the Deep BSDE method.

    Parameters
    ----------
    parameters : dict
        A dictionary containing the necessary parameters to solve
        the Backward Stochastic Differential Equation.
        Required parameters are: Q, R, S, A, B, C, D, G, L,
        x_0, batch_size, iteration_steps, N, lr_pi, and lr_gamma.

    Methods
    -------
    __init__(self, parameters)
        Initialize the BSDE_Solver class with the given parameters.

    neural_network(self, hidden_layers=4, isgamma=False)
        Create a neural network with uniform weight initialization.

    Euler_Maruyama_BSDE(self, dW)
        Performs the Euler-Maruyama scheme to solve the Backward Stochastic
        Differential Equation (BSDE) using the Deep BSDE method.

    Euler_Maruyama_control(self, dW)
        Perform the Euler-Maruyama scheme to update the state variable X
        and control variable Z, and compute the losses.

    train(self, dispay_steps)
        Train the neural network model using the Deep BSDE method.
        Stores the control and bsde losses in the class.
    """

    def __init__(self, parameters):
        """
        Initialize the BSDE_Solver class with the given parameters.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the necessary parameters to solve
            the Backward Stochastic Differential Equation.
             Required parameters are: Q, R, S, A, B, C, D, G, L,
             x_0, batch_size, iteration_steps, N, lr_pi, and lr_gamma.
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
                                                 minval=-0.5,
                                                 maxval=0.5))
        self.Z_0 = tf.Variable(tf.random.uniform((2, 1),
                                                 minval=-0.5,
                                                 maxval=0.5))

        # Set batch size, number of iterations and N
        self.batch_size = parameters['batch_size']
        self.iteration_steps = parameters['iteration_steps']
        self.N = parameters['N']

        # Set learning rates
        self.lr_pi = parameters['lr_pi']
        self.lr_gamma = parameters['lr_gamma']

        # Create N neural networks for pi and for gamma
        self.pi_networks = [
            self.neural_network(isgamma=False) for _ in range(self.N)
        ]
        self.gamma_networks = [
            self.neural_network(isgamma=True) for _ in range(self.N)
        ]

        # Set optimizers for each loss function
        self.optimizers_pi = [
            tf.keras.optimizers.Adam(self.lr_pi) for _ in range(self.N)
        ]
        self.optimizer_gamma = tf.keras.optimizers.Adam(self.lr_gamma)

    def neural_network(self, hidden_layers=4, isgamma=False):
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
                    minval=-1.0, maxval=1.0
                ),
                input_shape=(2, 1)),
        ])

        # Add `hidden_layers` - 1 hidden layers with ReLU activation
        for i in range(hidden_layers - 1):
            nn.add(tf.keras.layers.Dense(
                12, activation='relu',
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-1.0, maxval=1.0
                )))

        nn.add(tf.keras.layers.Dense(
            output_size,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-1.0, maxval=1.0
            )))
        nn.add(tf.keras.layers.Reshape((2, output_size)))

        return nn

    def Euler_Maruyama_BSDE(self, dW):
        """
        Performs the Euler-Maruyama scheme to solve the Backward Stochastic
        Differential Equation (BSDE) using the Deep BSDE method.

        Given a stochastic process `dW`, the function initializes
        `X`, `V`, and `Z` tensors with the provided initial values
        `x_0`, `V_0`, and `Z_0`, respectively. It then iterates over
        `N` time steps to compute the values of `V`, `Z`, and `X`
        using the Euler-Maruyama scheme.
        Finally, the function computes the loss using
        the last values of `X`, `V`, and `Z`.

        Parameters
        ----------
        dW : tensor of shape (batch_size, N, 1)
            The stochastic process used in the Euler-Maruyama scheme.

        Returns
        -------
        loss_1 : float
            The computed loss using the last values of `X`, `V`, and `Z`.
        """
        # Initialize X, V and Z
        X = tf.tile(
            tf.expand_dims(self.x_0, 0), [self.batch_size, 1, 1])
        V = tf.tile(
            tf.expand_dims(self.V_0, 0), [self.batch_size, 1, 1])
        Z = tf.tile(
            tf.expand_dims(self.Z_0, 0), [self.batch_size, 1, 1])

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
            Z = (
                Z - (1 / self.N) * self.functions.D_H(X, pi, Z, q)
                + q * dW_i
            )

            # Update X_i
            X = (
                X + self.functions.b(X, pi) * (1 / self.N)
                + self.functions.sigma(X, pi) * dW_i
            )

        # Compute the loss using the last values of X, V, Z
        loss_1 = tf.reduce_mean(
            tf.square(V + self.functions.g(X)) +
            0.5 * tf.reduce_sum(tf.square(Z + self.functions.D_g(X)),
                                axis=1)
        )

        return loss_1

    def Euler_Maruyama_control(self, dW):
        """
        Perform the Euler-Maruyama scheme to update the state variable X
        and control variable Z, and compute the losses.

        Parameters
        ----------
        dW : tensor of shape (N, batch_size, 1)
            The increments of the Brownian motion.

        Returns
        -------
        losses : list of N tensors, each of shape ()
            The losses for each time step, computed as the mean
            of the sum of squares of the partial derivatives of
            the value function with respect to the control variable.
        """
        losses = [0 for _ in range(self.N)]

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
            Z = (
                Z - (1 / self.N) * self.functions.D_H(X, pi, Z, q)
                + q * dW_i
            )

            # Update X_i
            X = (
                X + self.functions.b(X, pi) * (1 / self.N)
                + self.functions.sigma(X, pi) * dW_i
            )

            # Compute loss_i
            losses[i] += (
                tf.reduce_mean(
                    tf.reduce_sum(
                        tf.square(
                            self.functions.D_F(X, pi, Z, gamma)
                        ),
                        axis=1
                    )
                )
            )

        return losses

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
        bsde_losses = np.zeros(self.iteration_steps)
        control_losses = np.zeros(self.iteration_steps)

        # Record the start time of the iteration step
        start_time = time.time()
        print("Training in progress")
        print("-"*50)

        # Iteration steps
        for iteration_step in range(1, self.iteration_steps + 1):

            # Modify learning rates at specified intervals
            # if iteration_step % 200 == 0:

            #    self.optimizer_gamma.learning_rate.assign(
            #        self.optimizer_gamma.learning_rate.numpy() / 10
            #    )
            #    for i in range(self.N):
            #        self.optimizers_pi[i].learning_rate.assign(
            #            self.optimizers_pi[i].learning_rate.numpy() / 10
            #        )

            # Define Brownian motion with shape N x batch_size
            dW = (
                    tf.math.sqrt(1 / self.N) *
                    tf.random.normal(shape=(self.N, self.batch_size))
                )

            # Optimising the Gamma Parameters & V_0, Z_0
            with tf.GradientTape() as tape:

                # Watch V_0 and Z_0
                tape.watch([self.V_0, self.Z_0])

                # Define training parameters
                trainable_param = [self.V_0, self.Z_0]
                for i in range(self.N):
                    trainable_param += self.gamma_networks[i].trainable_weights

                # Compute BSDE loss for this iteration
                loss_1 = self.Euler_Maruyama_BSDE(dW)

            # Store BSDE loss from this iteration step
            bsde_losses[iteration_step-1] = loss_1.numpy()

            # Compute the gradients
            grads = tape.gradient(loss_1, trainable_param)

            # Use ADAM for optimisation
            self.optimizer_gamma.apply_gradients(zip(grads, trainable_param))

            # Optimising the Control Parameters
            with tf.GradientTape(persistent=True) as tape:

                # Define trainable parameters
                trainable_params = [0 for _ in range(self.N)]
                for i in range(self.N):
                    trainable_params[i] = (
                        self.pi_networks[i].trainable_weights
                    )

                # Compute losses
                losses = self.Euler_Maruyama_control(dW)

            # Compute the loss of the iteration as the sum of the losses
            loss_2 = tf.reduce_mean(tf.stack(losses))

            # Add the loss from this iteration to the list
            control_losses[iteration_step - 1] = loss_2

            # Update control parameters
            for i in range(self.N):

                # Define gradients
                gradients_i = tape.gradient(
                    losses[i], trainable_params[i]
                )

                # Apply gradients
                self.optimizers_pi[i].apply_gradients(
                    zip(gradients_i, trainable_params[i])
                )

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
