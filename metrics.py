
import torch
from abc import ABC, abstractmethod


class IQMetric(ABC):
    """Abstract IQ metric class.
    """
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class FullReferenceIQMetric(IQMetric):
    """Abstract class to implement full-reference IQ metrics.
    """

    @abstractmethod
    def __call__(self, im: torch.Tensor, im_ref: torch.Tensor, *args) -> torch.Tensor:
        """Compute the metric over im and

        :param im: Batch of distorted images. Size = N x C x H x W
        :param im_ref: Batch of reference images. Size = N x C x H x W
        :return: IQ metric for each pair. Size = N
        """
        raise NotImplementedError


class NoReferenceIQMetric(IQMetric):
    """Abstract class to implement no-reference IQ metrics.
    """

    @abstractmethod
    def __call__(self, im: torch.Tensor, *args) -> torch.Tensor:
        """Compute the metric over im and

        :param im: Batch of distorted images. Size = N x C x H x W
        :return: IQ metric for each pair. Size = N
        """
        raise NotImplementedError


class NormalizedMeanAbsoluteError(FullReferenceIQMetric):
    """Compute normalized mean absolute error (MAE) on images.

    Note that nMAE is a distortion metric, not a quality metric. This means that it should be negatively
    correlated with Mean Opinion Scores.
    """
    def __init__(self, norm=255.):
        super(NormalizedMeanAbsoluteError, self).__init__(name="nMAE")
        self.norm = norm

    def __call__(self, im: torch.Tensor, im_ref: torch.Tensor, *args) -> torch.Tensor:
        return torch.mean(torch.abs(im - im_ref) / self.norm, dim=[1, 2, 3])  # Average over C x H x W


# Aliases
# nMAE = NormalizedMeanAbsoluteError
