import tensorflow as tf


class BSDE_functions:
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
    f(X, pi) : The quadratic function f(X, pi).
    D_f(X, pi) : The derivative of f(X, pi) w.r.t. X.
    g(X) : The quadratic function g(X).
    D_g(X) : The derivative of the function g(X) w.r.t. X.
    b(X, pi) : The function b(X, pi).
    sigma(X, pi) : The function sigma(X, pi).
    H(X, pi, p, q) : The generalized Hamiltonian.
    D_H(X, pi, p, q) : The derivative of the generalized Hamiltonian w.r.t. X.
    F(X, pi, p, q) : The Hamiltonian F.
    D_F(X, pi, p, q) : The derivarive of the Hamiltonian F w.r.t. pi.
    """

    def __init__(self, parameters):
        """
        Initialize the BSDE_functions class with the given parameters.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters (Q, R, S, A, B, C, D, G, L)
            used in the functions.
        """

        self.Q = parameters['Q']
        self.R = parameters['R']
        self.S = parameters['S']
        self.A = parameters['A']
        self.B = parameters['B']
        self.C = parameters['C']
        self.D = parameters['D']
        self.G = parameters['G']
        self.L = parameters['L']

    def f(self, X, pi):
        """
        Compute the quadratic function f(X, pi):
            f(X, pi) = 1/2 X^T Q X + X^T S^T pi + 1/2 pi^T R pi

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).
        pi : tf.Tensor, shape (n, 2, 1)
            Input tensor pi, where n is the number of samples of shape (2, 1).

        Returns
        -------
        result : tf.Tensor, shape (n, 1, 1)
            The computed quadratic function value for each sample.
        """

        # Compute X^T and pi^T
        X_T = tf.transpose(X, perm=[0, 2, 1])
        pi_T = tf.transpose(pi, perm=[0, 2, 1])

        # Compute f(X, pi) = 1/2 X^T Q X + X^T S^T pi + 1/2 pi^T R pi
        f_value = (
            0.5 * tf.reduce_sum(tf.matmul(tf.matmul(X_T, self.Q), X),
                                axis=[1, 2]) +
            tf.reduce_sum(tf.matmul(tf.matmul(X_T, tf.transpose(self.S)), pi),
                          axis=[1, 2]) +
            0.5 * tf.reduce_sum(tf.matmul(tf.matmul(pi_T, self.R), pi),
                                axis=[1, 2])
        )

        # Reshape result to (n, 1, 1)
        return tf.reshape(f_value, (f_value.shape[0], 1, 1))

    def D_f(self, X, pi):
        """
        Compute the derivative of f(X, pi) w.r.t. X.
            D_f(X, pi) = Q X + S^T pi

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).
        pi : tf.Tensor, shape (n, 2, 1)
            Input tensor pi, where n is the number of samples of shape (2, 1).

        Returns
        -------
        result : tf.Tensor, shape (n, 2, 1)
            The computed derivative of f(X, pi) for each sample.

        """

        # Compute D_f(X, pi) = Q X + S^T pi
        D_f_value = (
            tf.matmul(self.Q, X) +
            tf.matmul(tf.transpose(self.S), pi)
        )

        return D_f_value

    def g(self, X):
        """
        Compute the quadratic function g(X).
            g(X) = 1/2 X^T G X + X^T L

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).

        Returns
        -------
        result : tf.Tensor, shape (n, 1, 1)
            The computed value of g(X) for each sample.
        """

        # Compute X^T
        X_T = tf.transpose(X, perm=[0, 2, 1])

        # Compite g(X) = 1/2 X^T G X + X^T L
        g_value = (
            0.5 * X_T @ self.G @ X +
            tf.matmul(X_T, self.L)
        )

        return g_value

    def D_g(self, X):
        """
        Compute the derivative of the function g(X) w.r.t. X.
            D_g(X) = G X + L

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).

        Returns
        -------
        result : tf.Tensor, shape (n, 2, 1)
            The computed derivative of g(X) for each sample.
        """

        # Compute D_g(X) = G X + L
        D_g_value = (
            tf.matmul(self.G, X) +
            self.L
        )

        return D_g_value

    def b(self, X, pi):
        """
        Compute the function b(X, pi).
            b(X, pi) = A X + B pi

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).
        pi : tf.Tensor, shape (n, 2, 1)
            Input tensor pi, where n is the number of samples of shape (2, 1).

        Returns
        -------
        result : tf.Tensor, shape (n, 2, 1)
            The computed b(X, pi) for each sample.
        """

        # Compute b(X, pi) = A X + B pi
        b_value = (
            tf.matmul(self.A, X) +
            tf.matmul(self.B, pi)
        )

        return b_value

    def sigma(self, X, pi):
        """
        Compute the function sigma(X, pi).
            sigma(X, pi) = C X + D pi

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).
        pi : tf.Tensor, shape (n, 2, 1)
            Input tensor pi, where n is the number of samples of shape (2, 1).

        Returns
        -------
        result : tf.Tensor, shape (n, 2, 1)
            The computed sigma(X, pi) for each sample.
        """

        # Compute sigma(X, pi) = C X + D pi
        sigma_value = (
            tf.matmul(self.C, X) +
            tf.matmul(self.D, pi)
        )

        return sigma_value

    def H(self, X, pi, p, q):
        """
        Compute the generalized Hamiltonian.
            H(X, pi, p, q) = b(X, pi)^T p + trace(sigma(X, pi)^T q) - f(X, pi).

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).
        pi : tf.Tensor, shape (n, 2, 1)
            Input tensor pi, where n is the number of samples of shape (2, 1).
        p : tf.Tensor, shape (n, 2, 1)
            Input tensor p where n is the number of samples of shape (2, 1).
        q : tf.Tensor, shape (n, 2, 1)
            Input tensor q where n is the number of samples of shape (2, 1).

        Returns
        -------
        result : tf.Tensor, shape (n, 1, 1)
            The computed H(X, pi, p, q) for each sample.
        """

        # Compute b(X, pi)^T and sigma(X, pi)^T
        b_T = tf.transpose(self.b(X, pi), perm=[0, 2, 1])
        sigma_T = tf.transpose(self.sigma(X, pi), perm=[0, 2, 1])

        # Compute H(X, pi, p, q)
        H_value = (
            tf.matmul(b_T, p) +
            tf.matmul(sigma_T, q) -
            self.f(X, pi)
        )

        return H_value

    def D_H(self, X, pi, p, q):
        """
        Compute the derivative of the generalized Hamiltonian w.r.t. X.
            D_H(X, pi, p, q) = A^T p + C^T q) - D_f(X, pi).

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).
        pi : tf.Tensor, shape (n, 2, 1)
            Input tensor pi, where n is the number of samples of shape (2, 1).
        p : tf.Tensor, shape (n, 2, 1)
            Input tensor p where n is the number of samples of shape (2, 1).
        q : tf.Tensor, shape (n, 2, 1)
            Input tensor q where n is the number of samples of shape (2, 1).

        Returns
        -------
        result : tf.Tensor, shape (n, 2, 1)
            The computed D_H(X, pi, p, q) for each sample.
        """

        # Compute D_H(X, pi, p, q) = A^T p + C^T q) - D_f(X, pi)
        D_H_value = (
            tf.matmul(tf.transpose(self.A), p) +
            tf.matmul(tf.transpose(self.C), q) -
            self.D_f(X, pi)
        )

        return D_H_value

    def F(self, X, pi, p, q):
        """
        Compute the Hamiltonian F:
            F(X, pi, p, q) = b(X, pi)^T p +
                             0.5*trace(sigma(X, pi) sigma(X, pi)^T q) -
                             f(X, pi)

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).
        pi : tf.Tensor, shape (n, 2, 1)
            Input tensor pi, where n is the number of samples of shape (2, 1).
        p : tf.Tensor, shape (n, 2, 1)
            Input tensor p where n is the number of samples of shape (2, 1).
        q : tf.Tensor, shape (n, 2, 2)
            Input tensor q where n is the number of samples of shape (2, 2).

        Returns
        -------
        result : tf.Tensor, shape (n, 1, 1)
            The computed F(X, pi, p, q) for each sample.
        """

        # Compute b(X, pi)^T and sigma(X, pi)^T
        b_T = tf.transpose(self.b(X, pi), perm=[0, 2, 1])
        sigma_T = tf.transpose(self.sigma(X, pi), perm=[0, 2, 1])

        # Compute F(X, pi, p, q)
        F_value = (
            tf.matmul(b_T, p) +
            0.5 * tf.matmul(tf.matmul(sigma_T, q), self.sigma(X, pi)) -
            self.f(X, pi)
        )

        return F_value

    def D_F(self, X, pi, p, q):
        """
        Compute the derivarive of the Hamiltonian F w.r.t. pi:
            D_F(X, pi, p, q) = B^T p + D^T q (C X + D pi) - S X - R pi

        Parameters
        ----------
        X : tf.Tensor, shape (n, 2, 1)
            Input tensor X, where n is the number of samples of shape (2, 1).
        pi : tf.Tensor, shape (n, 2, 1)
            Input tensor pi, where n is the number of samples of shape (2, 1).
        p : tf.Tensor, shape (n, 2, 1)
            Input tensor p where n is the number of samples of shape (2, 1).
        q : tf.Tensor, shape (n, 2, 2)
            Input tensor q where n is the number of samples of shape (2, 2).

        Returns
        -------
        result : tf.Tensor, shape (n, 2, 1)
            The computed F(X, pi, p, q) for each sample.
        """

        # Compute D_F(X, pi, p, q) = B^T p + D^T q (C X + D pi) - S X - R pi
        D_f_value = (
            tf.transpose(self.B) @ p +
            tf.transpose(self.D) @ q @ (self.C @ X + self.D @ pi) -
            self.S @ X - self.R @ pi
        )

        return D_f_value
