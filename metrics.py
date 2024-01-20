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


class IntersectionOverUnion(IQMetric):
    def __init__(self, num_classes):
        super(IntersectionOverUnion, self).__init__(name="IoU")
        self.num_classes = num_classes

    def __call__(self, y_pred, y_true):
        intersection = torch.logical_and(y_pred, y_true).sum(dim=[1, 2, 3])
        union = torch.logical_or(y_pred, y_true).sum(dim=[1, 2, 3])
        iou = intersection / union
        return iou


class DiceCoefficient(IQMetric):
    def __init__(self):
        super(DiceCoefficient, self).__init__(name="Dice")

    def __call__(self, y_pred, y_true):
        intersection = torch.logical_and(y_pred, y_true).sum(dim=[1, 2, 3])
        dice = (2 * intersection) / (y_pred.sum(dim=[1, 2, 3]) + y_true.sum(dim=[1, 2, 3]))
        return dice


class PixelAccuracy(IQMetric):
    def __init__(self):
        super(PixelAccuracy, self).__init__(name="PixelAccuracy")

    def __call__(self, y_pred, y_true):
        correct_pixels = (y_pred == y_true).sum(dim=[1, 2, 3])
        total_pixels = y_pred.numel()
        pixel_accuracy = correct_pixels / total_pixels
        return pixel_accuracy


class ClassAccuracy(IQMetric):
    def __init__(self, num_classes):
        super(ClassAccuracy, self).__init__(name="ClassAccuracy")
        self.num_classes = num_classes

    def __call__(self, y_pred, y_true):
        class_accuracy = []
        for i in range(self.num_classes):
            correct_pixels = ((y_pred == i) & (y_true == i)).sum(dim=[1, 2, 3])
            total_pixels = (y_true == i).sum(dim=[1, 2, 3])
            class_accuracy.append(correct_pixels / total_pixels)
        return class_accuracy

# Aliases
# nMAE = NormalizedMeanAbsoluteError
# IoU = IntersectionOverUnion
# Dice = DiceCoefficient
# PA = PixelAccuracy
# CA = ClassAccuracy
