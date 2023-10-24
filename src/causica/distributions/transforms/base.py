"""
Wrapper around torch.distributions.transforms to allow for joint transforms on TensorDicts.
"""
import weakref
from typing import Any, Generic, Optional, TypeVar, Union

import torch
from torch import nn
from torch.distributions.transforms import Transform, _InverseTransform

T_co = TypeVar("T_co", covariant=True, bound="TypedTransform")


class _TransformRef(Generic[T_co]):
    """A covariant reference to a transform.

    Particularily used to allow subclasses to refine the inverse type.
    """

    weak_ref: Optional[weakref.ReferenceType[T_co]]

    def __init__(self, obj: T_co):
        self.weak_ref = weakref.ref(obj)

    def __call__(self) -> Optional[T_co]:
        return self.weak_ref() if self.weak_ref is not None else None


X = TypeVar("X", covariant=False)
Y = TypeVar("Y", covariant=False)


class TypedTransform(Generic[X, Y], Transform):
    """Transforms with typed argument and return values.

    Should not be instantiated directly, but rather through the `TypedTransform` class.

    Notes:
        Special care needs to be taken in overriding the inverse. In particular, the weak reference is of a
        non-covariant generic type, and therefore isn't compatible with subclasses returning refined inverses. We solve
        this by using a custom covariant reference class.
    """

    def __call__(self, x: X) -> Y:  # pylint: disable=useless-parent-delegation
        return super().__call__(x)  # type: ignore

    def _inv_call(self, y: Y) -> X:  # pylint: disable=useless-parent-delegation
        return super()._inv_call(y)  # type: ignore

    def _call(self, x: X) -> Y:  # pylint: disable=useless-parent-delegation
        return super()._call(x)

    def _inverse(self, y: Y) -> X:  # pylint: disable=useless-parent-delegation
        return super()._inverse(y)

    def log_abs_det_jacobian(self, x: X, y: Y):  # pylint: disable=useless-parent-delegation
        return super().log_abs_det_jacobian(x, y)

    @property
    def inv(self) -> "TypedTransform[Y, X]":
        inv: Optional[TypedTransform[Y, X]] = None
        match self._inv:
            case _TransformRef():
                inv = self._inv()
            case TypedTransform():
                inv = self._inv
        if inv is None:
            inv = _TypedInverseTransform[Y, X](self)
            self._inv: Union[_TransformRef[_TypedInverseTransform[Y, X]], TypedTransform[Y, X]] = _TransformRef[
                _TypedInverseTransform[Y, X]
            ](inv)
        return inv


class _TypedInverseTransform(Generic[Y, X], TypedTransform[Y, X], _InverseTransform):
    _inv: TypedTransform[X, Y]


class TransformModule(Generic[X, Y], TypedTransform[X, Y], nn.Module):
    """Transforms with learnable parameters.

    This is similar to the `pyro.distributions.torch_transform.TransformModule` class.
    """

    def __hash__(self):
        """Return the nn.Module based hash.

        Notes:
            The Transformation hash is None.
        """
        return super(torch.nn.Module, self).__hash__()

    @property
    def inv(self) -> "TransformModule[Y, X]":
        inv: Optional[TransformModule[Y, X]] = None
        match self._inv:
            case _TransformRef():
                inv = self._inv()
            case TransformModule():
                inv = self._inv
        if inv is None:
            inv = _InverseTransformModule[Y, X](self)
            self._inv: Union[_TransformRef[_InverseTransformModule[Y, X]], TransformModule[Y, X]] = _TransformRef[
                _InverseTransformModule[Y, X]
            ](inv)
        return inv


class _InverseTransformModule(Generic[Y, X], TransformModule[Y, X], _TypedInverseTransform):
    """Inverse transformation of a TransformModule."""

    _inv: TransformModule[X, Y]


class SequentialTransformModule(Generic[X, Y], TransformModule[X, Y], nn.Sequential):
    """Sequential transform with TransformModule transformations."""

    def __init__(self, *args: TransformModule[X, Y], cache_size: int = 0):
        super().__init__(cache_size=cache_size)
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def _inverse(self, y: Y) -> X:
        current: Any = y
        # Create a tuple of submodules before reversing, to avoid a copy of the Sequential module
        for module in tuple(self)[::-1]:
            assert isinstance(module, TransformModule)
            current = module.inv(current)
        return current

    def _call(self, x: X) -> Y:
        return nn.Sequential.__call__(self, x)
