import abc
from . import AbstractMachine
import numpy as _np


class AbstractDensityMatrix(AbstractMachine):
    """Abstract class for NetKet density matrices"""

    def __init__(self, hilbert):
        super().__init__()
        self.hilbert = hilbert

    @abc.abstractmethod
    def log_val(self, xr, xc=None, out=None):
        r"""Computes the logarithm of the density matrix for a batch of visible
        quantum numbers `(xr,xc)` and stores the result into `out`.
        Specifically, for each element of the batch i, this function should compute
        out[i]=log(rho(xr[i],xc[i])).
        If xr is None, it is assumed that xr has twice as many quantum numbers and
        contains both row and columns, stored contigously.

        Args:
            xr: A matrix of `float64` of shape `(*, self.n_visible)` if xc is given.
                If xc is None, then this should be a matrix of `float64` of shape `(*, 2*self.n_visible)`.
            xc (optional): A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. The length of `out` should be `xr.shape[0]`.

        Returns:
            A vector out[i]=log(rho(xr[i],xc[i])).
        """
        raise NotImplementedError

    def der_log(self, xr, xc=None, out=None):
        r"""Computes the gradient of the logarithm of the density matrix for a
        batch of visible configurations `(xr,xc)` and stores the result into `out`.

        Args:
            xr: A matrix of `float64` of shape `(*, self.n_visible)` if xc is given.
                If xc is None, then this should be a matrix of `float64` of shape `(*, 2*self.n_visible)`.
            xc (optional): A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`.
                `out` should be a matrix of shape `(x.shape[0], self.n_par)`.

        Returns:
            `out`
        """
        raise NotImplementedError

    def diagonal(self):
        r"""Returns a view of this density matrix as a Machine computing log_val(x)=rho(x,x).

        Returns:
            `diagonal`
        """

    def to_matrix(self, normalize=True):
        r"""
        Returns a numpy array representation of the density matrix.
        Note that, in general, the size of the matrix is exponential
        in the number of quantum numbers, and this operation should thus
        only be performed for low-dimensional Hilbert spaces.

        This method requires an indexable Hilbert space.

        Args:
            normalize (bool): If True, the returned matrix is normalized in such a way that Tr(rho) =1.

        Returns:
            numpy.array: The matrix rho(x,x') for all states x,x' in the Hilbert space.

        """
        if self.hilbert.is_indexable:
            raise NotImplementedError("to_matrix to be implemented")

        else:
            raise RuntimeError("The hilbert space is not indexable")

    def log_trace(self):
        r"""
        Returns the log of the trace of the density matrix.
        Note that, in general, the size of the density matrix is exponential
        in the number of quantum numbers, and this operation should thus
        only be performed for low-dimensional Hilbert spaces.

        This method requires an indexable Hilbert space.

        Returns:
            float: log(Tr(rho))

        """
        if self.hilbert.is_indexable:
            all_states = self.hilbert.all_states()
            rho_diag = self.log_val(all_states, all_states).real
            maxl = rho_diag.max()
            log_t = _np.log(_np.exp((rho_diag - maxl).sum()))

            return log_t + maxl
        else:
            raise RuntimeError("The hilbert space is not indexable")

    @property
    def input_size(self):
        return 2 * self.hilbert.size
