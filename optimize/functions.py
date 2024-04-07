import torch as tr
import numpy as np

def rosen(x):
    """
    The Rosenbrock function.

    The function computed is::

        sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be computed.

    Returns
    -------
    f : float
        The value of the Rosenbrock function.

    See Also
    --------
    rosen_der, rosen_hess, rosen_hess_prod

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen
    >>> X = 0.1 * np.arange(10)
    >>> rosen(X)
    76.56
    

    For higher-dimensional input ``rosen`` broadcasts.
    In the following example, we use this to plot a 2D landscape.
    Note that ``rosen_hess`` does not broadcast in this manner.

    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> x = np.linspace(-1, 1, 50)
    >>> X, Y = np.meshgrid(x, x)
    >>> ax = plt.subplot(111, projection='3d')
    >>> ax.plot_surface(X, Y, rosen([X, Y]))
    >>> plt.show()
    """
    x = tr.as_tensor(x)
    r = tr.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
                  axis=0)
    return r

def rosen_der(x):
    """
    The derivative (i.e. gradient) of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the derivative is to be computed.

    Returns
    -------
    rosen_der : (N,) ndarray
        The gradient of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_hess, rosen_hess_prod

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen_der
    >>> X = 0.1 * np.arange(9)
    >>> rosen_der(X)
    array([ -2. ,  10.6,  15.6,  13.4,   6.4,  -3. , -12.4, -19.4,  62. ])

    """
    x = np.asarray(x)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = (200 * (xm - xm_m1**2) -
                 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der

def rosen_hess(x):
    """
    The Hessian matrix of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Hessian matrix is to be computed.

    Returns
    -------
    rosen_hess : ndarray
        The Hessian matrix of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_der, rosen_hess_prod

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen_hess
    >>> X = 0.1 * np.arange(4)
    >>> rosen_hess(X)
    array([[-38.,   0.,   0.,   0.],
           [  0., 134., -40.,   0.],
           [  0., -40., 130., -80.],
           [  0.,   0., -80., 200.]])

    """
    x = np.atleast_1d(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros(len(x), dtype=x.dtype)
    diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H