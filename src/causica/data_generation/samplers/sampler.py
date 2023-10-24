import abc
from typing import Generic, TypeVar

SampleType = TypeVar("SampleType")


class Sampler(Generic[SampleType], abc.ABC):
    """
    An interface of a sampler, useful for generative processes

    The interface only allows sampling one thing at a time.
    """

    @abc.abstractmethod
    def sample(self) -> SampleType:
        """Sample a sample type with given shape"""
