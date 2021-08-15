# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union
from functools import partial

import jax
from jax import numpy as jnp
from flax import struct

import netket.jax as nkjax
from netket.utils.types import PyTree
from netket.utils import warn_deprecation

from .qgt_onthefly_logic import mat_vec_factory

from ..linear_operator import LinearOperator, Uninitialized


def QGTOnTheFly(vstate=None, **kwargs) -> "QGTOnTheFlyT":
    """
    Lazy representation of an S Matrix computed by performing 2 jvp
    and 1 vjp products, using the variational state's model, the
    samples that have already been computed, and the vector.
    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    Args:
        vstate: The variational State.
    """
    if vstate is None:
        return partial(QGTOnTheFly, **kwargs)

    if "centered" in kwargs:
        warn_deprecation(
            "The argument `centered` is deprecated. The implementation now always behaves as if centered=False."
        )
        kwargs.pop("centered")

    if jnp.ndim(vstate.samples) == 2:
        samples = vstate.samples
    else:
        samples = vstate.samples.reshape((-1, vstate.samples.shape[-1]))

    mat_vec = mat_vec_factory(
        forward_fn=vstate._apply_fun,
        params=vstate.parameters,
        model_state=vstate.model_state,
        samples=samples,
    )

    return QGTOnTheFlyT(
        mat_vec=mat_vec,
        params=vstate.parameters,
        **kwargs,
    )


@struct.dataclass
class QGTOnTheFlyT(LinearOperator):
    """
    Lazy representation of an S Matrix computed by performing 2 jvp
    and 1 vjp products, using the variational state's model, the
    samples that have already been computed, and the vector.
    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    mat_vec: Callable[[PyTree, float], PyTree] = Uninitialized
    """The S matrix-vector product as generated by mat_vec_factory.
    It's a jax.Partial, so can be used as pytree_node."""

    params: PyTree = Uninitialized
    """The first input to apply_fun (parameters of the ansatz). 
    Only used as a shape placeholder."""

    def __matmul__(self, y):
        return onthefly_mat_treevec(self, y)

    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree], **kwargs) -> PyTree:
        return _solve(self, solve_fun, y, x0=x0)

    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.
        Returns:
            A dense matrix representation of this S matrix.
        """
        return _to_dense(self)


@jax.jit
def onthefly_mat_treevec(
    S: QGTOnTheFly, vec: Union[PyTree, jnp.ndarray]
) -> Union[PyTree, jnp.ndarray]:
    """
    Perform the lazy mat-vec product, where vec is either a tree with the same structure as
    params or a ravelled vector
    """

    # if hasa ndim it's an array and not a pytree
    if hasattr(vec, "ndim"):
        if not vec.ndim == 1:
            raise ValueError("Unsupported mat-vec for batches of vectors")
        # If the input is a vector
        if not nkjax.tree_size(S.params) == vec.size:
            raise ValueError(
                """Size mismatch between number of parameters ({nkjax.tree_size(S.params)})
                                and vector size {vec.size}.
                             """
            )

        _, unravel = nkjax.tree_ravel(S.params)
        vec = unravel(vec)
        ravel_result = True
    else:
        ravel_result = False

    vec = nkjax.tree_cast(vec, S.params)

    res = S.mat_vec(vec, S.diag_shift)

    if ravel_result:
        res, _ = nkjax.tree_ravel(res)

    return res


@jax.jit
def _solve(
    self: QGTOnTheFlyT, solve_fun, y: PyTree, *, x0: Optional[PyTree], **kwargs
) -> PyTree:

    y = nkjax.tree_cast(y, self.params)

    # we could cache this...
    if x0 is None:
        x0 = jax.tree_map(jnp.zeros_like, y)

    out, info = solve_fun(self, y, x0=x0)
    return out, info


@jax.jit
def _to_dense(self: QGTOnTheFlyT) -> jnp.ndarray:
    """
    Convert the lazy matrix representation to a dense matrix representation
    Returns:
        A dense matrix representation of this S matrix.
    """
    Npars = nkjax.tree_size(self.params)
    I = jax.numpy.eye(Npars)
    out = jax.vmap(lambda x: self @ x, in_axes=0)(I)

    if nkjax.is_complex(out):
        out = out.T

    return out
