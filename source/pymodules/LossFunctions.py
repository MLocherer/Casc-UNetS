r"""
    Copyright (C) 2022  Mark Locherer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
sys.path.append("pymodules")
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor




def confusion_matrix(pred: Tensor, target: Tensor, num_classes: int, batch: bool = False, target_one_hot = False) -> tuple:
    r""" Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e., the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)

    Parameters:
        num_classes: the number of classes
        pred: Prediction tensor in one hot encoding
        target: Groundtruth tensor in one hot encoding
        batch: if batch is set to False: prediction and groundtruth are both tensors of shape C, H, W where C are the image
        channels and H, W are the hight and width of the image.
    Return:
        The function returns a tuple of tensors with contains the true positives, false pos., true negatives, and
        false negatives for each channel.
    """

    if target_one_hot:
        # Error handling check dimensions and reduction dimension setting for torch.sum()
        if not batch:
            assert (len(pred.shape) == 3) and (pred.shape == target.shape), f"pred / gt must have shape C x H x W " \
                                                                            f"pred: {pred.shape} gt: {target.shape}"
            reduce_dim = (1, 2)
        else:
            assert (len(pred.shape) == 4) and (pred.shape == target.shape), f"pred / gt must have shape N x C x H x W " \
                                                                            f"but has shape: pred: {pred.shape} " \
                                                                            f"gt: {target.shape}"
            reduce_dim = (2, 3)
    else:
        # target is label encoded
        # Error handling check dimensions and reduction dimension setting for torch.sum()
        if not batch:
            assert (len(pred.shape) == 3) and (len(target.shape) == 2), f"prediction must have shape C x H x W " \
                                                                    f"but has shape:  {pred.shape} " \
                                                                    f"groundtruth must have shape H x W " \
                                                                    f"but has shape: {target.shape} "
            # convert class label to one hot encoding
            target = F.one_hot(target, num_classes=num_classes).permute(2, 0, 1)
            reduce_dim = (1, 2)
        else:
            assert (len(pred.shape) == 4) and (len(target.shape) == 3), f"prediction must have shape N x C x H " \
                                                                    f" x W but has shape:  " \
                                                                    f" {pred.shape} groundtruth must " \
                                                                    f" have shape N x H x W but has shape: " \
                                                                    f"{target.shape}"
            # convert class label to one hot encoding
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
            reduce_dim = (2, 3)

        # the shapes of pred and gt do not match
        assert pred.shape == target.shape, f"the shapes of the prediction tensor {pred.shape} and the groundtruth tensor {target.shape} do not match, after one-hot encoding and permutation. \n pred: {pred.cpu().numpy()} \n gt: {target.cpu().numpy()}"

    true_positives = torch.sum(pred * target, reduce_dim)
    # false positives are prediction inputt positives - true positives
    false_positives = torch.sum(pred, reduce_dim) - true_positives
    # false negatives are the target positives - the true positives
    false_negatives = torch.sum(target, reduce_dim) - true_positives

    total_pixels = target.shape[reduce_dim[0]] * target.shape[reduce_dim[1]]
    true_negatives = total_pixels - true_positives - false_positives - false_negatives

    return true_positives, false_positives, true_negatives, false_negatives


def compute_class_weights(data_set: torch.utils.data.Dataset) -> Tensor:
    r"""
    Calculates the weights in the given Dataset object, to be used in conjunction w/ nn.CrossEntropyLoss. Uses the
    sklearn function compute_class_weight (https://sklearn.org/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
    data_loader labels must be class labels
    Args:
        data_set: Dataset object for which the weights have to be calculated

    Returns:
    Class weights for the given DataLoader object.
    Examples:
        testset = TeslaSiemensDataset(root_dir='data/siemens_reduced/test/', transform=transform_test, include_cap=True, num_of_surrouding_imgs=1)
        class_weights = find_class_weights(testset)
        # returns tensor([139.5182,  27.2220,  95.0956,   0.2534])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    """
    # create a list w/ all targets to calculate the number for each class
    # convert first from one_hot to class
    t_list = [t.flatten() for _, t in data_set]
    ts = torch.cat(t_list, dim=0)
    class_weights = compute_class_weight('balanced', classes=ts.unique().numpy(), y=ts.numpy())

    return torch.Tensor(class_weights)


def count_abs_class_frequs(data_set: torch.utils.data.Dataset, num_classes: int) -> Tensor:
    r"""
    Returns the absolute frequency of each target class in the Dataset data_set.
    Args:
        data_set: Dataset for which the target classes have to be counted
        num_classes: number of classes

    Returns:
        Number of classes for each class in a Tensor object
    Examples:
        class_abs_frequ_train = count_class_frequs(trainset, 4)
        # returns: tensor([  107740,   402019,   316436, 62481581], dtype=torch.int32)
    """
    class_frequencies = torch.zeros(num_classes, dtype=torch.int32)
    for i in range(len(data_set)):
        class_frequencies += data_set[i][1].flatten().bincount(minlength=num_classes)
    return class_frequencies


def count_rel_class_frequs(data_set: torch.utils.data.Dataset, num_classes: int) -> Tensor:
    r"""
        Returns the relative frequency of each target class in the Dataset data_set.
        Args:
            data_set: Dataset for which the target classes have to be counted
            num_classes: number of classes

        Returns:
            Relative frequency of classes for each class in a Tensor object
        Examples:
            class_abs_frequ_train = count_class_frequs(trainset, 4)
            # returns: tensor([0.0017, 0.0064, 0.0050, 0.9869])
            # Cap class: 0.0017, ..., bg class 0.9869
        """
    class_frequs = count_abs_class_frequs(data_set, num_classes)
    return class_frequs / torch.sum(class_frequs)

# ----------------------------------------------------------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------------------------------------------------------


class BatchCrossEntropyLoss(nn.Module):
    r"""
    Computes the CrossEntropyLoss of input and target. input is a Tensor of shape [N x C x H x W], target is a tensor
    of shape [N x H x W]. This class uses the pytorch function cross_entropy.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=-100):
        super(BatchCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        '''
        args: input: tensor of shape (N, C, H, W)
        args: target: tensor of shape(N, H, W)
        '''
        return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index)


class DiceLossEx(nn.Module):
    r"""
    DiceLoss class: Computes the dice loss b/w the inputt or prediction tensor input and the target or groundtruth
    tensor target.
    """

    def __init__(self, smooth=1e-5, weight=None, normalisation="sigmoid", exclude_classes: list=None, target_one_hot=False):
        r"""

        Args:
            smooth: smooth value to prevent div by zero
            weight: optional weighting
            normalisation: normatistaion function to be applied as the output layer of the model net
            exclude_classes: list of class indices to be excluded from the loss calculation
        """
        super(DiceLossEx, self).__init__()

        self.smooth = smooth

        if self.smooth < 0 or self.smooth > 1.0:
            raise ValueError("smooth value must be in [0,1]")

        self.weight = weight
        self.normalization = get_normalisation_fun(normalisation)
        self.exclude_classes = exclude_classes
        self.target_one_hot = target_one_hot

    def forward(self, inputt: Tensor, target: Tensor) -> Tensor:
        r"""
        forward method to compute the dice loss b/w inputt and target
        Args:
            inputt: prediction tensor
            target: target or groundtruth tensor

        Returns: dice loss
        """
        if self.target_one_hot:
            assert (len(inputt.shape) == 4) and (target.shape == inputt.shape), f"input / target must have shape " \
                                                                                f"N x C x H x W. input shape: " \
                                                                                f"{inputt.shape} " \
                                                                                f"target: {target.shape} "
            num_classes = target.shape[1]

        else:
            # label encoded
            assert (len(inputt.shape) == 4) and (len(target.shape) == 3), f"input must have shape N x C x H " \
                                                                          f" x W but has shape:  " \
                                                                          f" {inputt.shape} target must " \
                                                                          f" have shape N x H x W but has shape: " \
                                                                          f"{target.shape}"
            num_classes = inputt.shape[1]
            # create one hot encoding, derive number of classes from input
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # apply normalisation to input
        inputt = self.normalization(inputt)

        include_classes = list(range(0, num_classes))
        if self.exclude_classes is not None:
            include_classes = [c for c in include_classes if c not in self.exclude_classes]
        target = target[:, include_classes]
        inputt = inputt[:, include_classes]

        # intersection of each class are the true positives
        # e.g w/ 3 classes intersect tensor([[5., 1., 0.]])
        intersect = torch.sum(target * inputt, (2, 3))
        # sum of cardinalities of each set are 2 * true- + false positives + false negatives
        # e.g. cardi tensor([[14.,  9.,  6.]])
        cardi = torch.sum(target, (2, 3)) + torch.sum(inputt, (2, 3))
        # adjust w/ weights
        # e.g. weight = torch.tensor([1, 2, 3])
        if self.weight is not None:
            # tensor([[5., 2., 0.]])
            intersect = self.weight * intersect
            # cardi tensor([[14., 18., 18.]])
            cardi = self.weight * cardi

        # dice loss (sum over all classes)
        dice_loss = 1 - (2 * torch.sum(intersect) + self.smooth) / (torch.sum(cardi) + self.smooth)

        return dice_loss


class TverskyLoss(nn.Module):
    r"""
    TverskyLoss class: Computes the tversky loss b/w the inputt or prediction tensor input and the target or groundtruth
    tensor target. https://arxiv.org/abs/1810.07842
    """

    def __init__(self, smooth=1e-5, alpha=0.7, beta=0.3, normalisation="sigmoid", exclude_classes: list=None, target_one_hot=False):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth

        if self.smooth < 0 or self.smooth > 1.0:
            raise ValueError("smooth value should be in [0,1]")

        self.alpha = alpha
        self.beta = beta
        self.normalization = get_normalisation_fun(normalisation)
        self.exclude_classes = exclude_classes
        self.target_one_hot = target_one_hot

    def forward(self, inputt: Tensor, target: Tensor) -> Tensor:
        r"""
        forward method to compute the tversky index loss b/w inputt and target
        Args:
            inputt: prediction tensor
            target: target or groundtruth tensor

        Returns: tversky index loss
        """
        if self.target_one_hot:
            assert (len(inputt.shape) == 4) and (target.shape == inputt.shape), f"input / target must have shape " \
                                                                                f"N x C x H x W. input shape: " \
                                                                                f"{inputt.shape} " \
                                                                                f"target: {target.shape} "
            num_classes = target.shape[1]

        else:
            # label encoded
            assert (len(inputt.shape) == 4) and (len(target.shape) == 3), f"input must have shape N x C x H " \
                                                                          f" x W but has shape:  " \
                                                                          f" {inputt.shape} target must " \
                                                                          f" have shape N x H x W but has shape: " \
                                                                          f"{target.shape}"
            num_classes = inputt.shape[1]
            # create one hot encoding, derive number of classes from input
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # apply normalisation to input
        inputt = self.normalization(inputt)

        include_classes = list(range(0, num_classes))
        if self.exclude_classes is not None:
            include_classes = [c for c in include_classes if c not in self.exclude_classes]
        target = target[:, include_classes]
        inputt = inputt[:, include_classes]

        # intersection of each class are the true positives
        # e.g w/ 3 classes intersect tensor([[5., 1., 0.]])
        # true positive is the intersection of the prediction inputt and groundtruth, namely the
        # correctly predicted positives
        # true positives reduce over image dimensions H x W
        true_positives = torch.sum(target * inputt, (2, 3))
        # false positives are prediction inputt positives - true positives
        false_positives = torch.sum(inputt, (2, 3)) - true_positives
        # false negatives are the target positives - the true positives
        false_negatives = torch.sum(target, (2, 3)) - true_positives
        # Tversky index
        ti = (torch.sum(true_positives) + self.smooth) / (torch.sum(
            true_positives + self.alpha * false_positives + self.beta * false_negatives) + self.smooth)

        return 1 - ti


class FocalTverskyLoss(nn.Module):
    r"""
    FocalTverskyLoss class: Computes the focal tversky loss b/w the inputt or prediction tensor input and the target or groundtruth
    tensor target. γ varies in the range [1, 3]. In practice, if a pixel is misclassified with a high Tversky index, the FTL is unaf-
    fected. However, if the Tversky index is small and the pixel is misclassified, the FTL will decrease significantly. When γ > 1,
    the loss function focuses more on less accurate predictions that have been misclassified. https://arxiv.org/abs/1810.07842
    """

    def __init__(self, smooth=1e-5, alpha=0.7, beta=0.3, gamma=4 / 3, normalisation="sigmoid", exclude_classes: list=None, target_one_hot=False):
        super(FocalTverskyLoss, self).__init__()

        self.smooth = smooth

        if self.smooth < 0 or self.smooth > 1.0:
            raise ValueError("smooth value should be in [0, 1]")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if self.gamma < 0 or self.gamma > 3.0:
            raise ValueError("smooth value should be in [1, 3]")

        self.normalization = get_normalisation_fun(normalisation)
        self.exclude_classes = exclude_classes
        self.target_one_hot = target_one_hot

    def forward(self, inputt: Tensor, target: Tensor) -> Tensor:
        r"""
        forward method to compute the tversky index loss b/w inputt and target
        Args:
            inputt: prediction tensor
            target: target or groundtruth tensor

        Returns: tversky index loss

        """
        if self.target_one_hot:
            assert (len(inputt.shape) == 4) and (target.shape == inputt.shape), f"input / target must have shape " \
                                                                                f"N x C x H x W. input shape: " \
                                                                                f"{inputt.shape} " \
                                                                                f"target: {target.shape} "
            num_classes = target.shape[1]

        else:
            # label encoded
            assert (len(inputt.shape) == 4) and (len(target.shape) == 3), f"input must have shape N x C x H " \
                                                                          f" x W but has shape:  " \
                                                                          f" {inputt.shape} target must " \
                                                                          f" have shape N x H x W but has shape: " \
                                                                          f"{target.shape}"
            num_classes = inputt.shape[1]
            # create one hot encoding, derive number of classes from input
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # apply normalisation to input
        inputt = self.normalization(inputt)

        include_classes = list(range(0, num_classes))
        if self.exclude_classes is not None:
            include_classes = [c for c in include_classes if c not in self.exclude_classes]
        target = target[:, include_classes]
        inputt = inputt[:, include_classes]

        # intersection of each class are the true positives
        # e.g w/ 3 classes intersect tensor([[5., 1., 0.]])
        # true positive is the intersection of the prediction inputt and groundtruth, namely the
        # correctly predicted positives
        # true positives reduce over image dimensions H x W
        true_positives = torch.sum(target * inputt, (2, 3))
        # false positives are prediction inputt positives - true positives
        false_positives = torch.sum(inputt, (2, 3)) - true_positives
        # false negatives are the target positives - the true positives
        false_negatives = torch.sum(target, (2, 3)) - true_positives

        # Tversky index
        ti = (torch.sum(true_positives) + self.smooth) / (torch.sum(
            true_positives + self.alpha * false_positives + self.beta * false_negatives) + self.smooth)

        return torch.pow(1 - ti, 1 / self.gamma)


# ----------------------------------------------------------------------------------------------------------------------
# Activation functions
# ----------------------------------------------------------------------------------------------------------------------

class Threshold(nn.Module):
    r"""
    Alternative activation function for multi-class pixelwise semantic segmentation.
    """
    def __init__(self, threshold=.5):
        super().__init__()
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Threshold value must be in [0,1]")
        else:
            self.threshold = threshold

    def min_max_fscale(self, x):
        r"""
        Applies min max feature scaling to input. Each channel is treated individually.
        Input is assumed to be N x C x H x W (one-hot-encoded prediction)
        """
        x_f = x.flatten(2)
        x_min, x_max = x_f.min(-1, True).values, x_f.max(-1, True).values

        x_f = (x_f - x_min) / (x_max - x_min)
        return x_f.reshape_as(x)

    def forward(self, input):
        assert (len(input.shape) == 4), f"input has wrong number of dims. Must have dim = 4 but has dim {input.shape}"

        input = self.min_max_fscale(input)

        m = nn.Threshold(threshold=self.threshold, value=0.0)
        return m(input)


class Identity(nn.Module):
    r"""
    Identity activation function f(x) = x
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------

def get_normalisation_fun(normalisation_name: str):
    r"""
    return an activation layer object or normalisation function.
    Args:
        normalisation_name: string name for normalisation

    Returns:

    """
    possible_normalisations = ["sigmoid", "softmax", "identity", 'threshold']
    assert normalisation_name in possible_normalisations, "unknown normalisation specified, allowed types " \
                                                                "are: sigmoid, softmax, or identity "
    if normalisation_name == "sigmoid":
        norm = nn.Sigmoid()
    elif normalisation_name == "softmax":
        norm = nn.Softmax(dim=1)
    elif normalisation_name == "identity":
        norm = Identity()
    elif normalisation_name == "threshold":
        norm = Threshold(threshold=.5)

    return norm


def get_loss_fun(loss_fun_name: str, normalisation_name: str, weights: torch.Tensor=None, exclude_classes: list=None,
                 target_one_hot: bool = False):
    r"""
    Returns a loss function object for the given parameters
    Args:
        target_one_hot: target one hot version of loss function
        loss_fun_name: Name of loss function
        normalisation_name: Name of normalisation
        weights: weights for the classes
        exclude_classes: list of classes to exclude, the first index is chosen as ignore index for CE and WCE

    Returns:

    """
    if weights is None and loss_fun_name in ['GDL', 'WCE']:
        assert False, f"weights must be specified for the given loss fun {loss_fun_name}"

    if loss_fun_name == "DL":
        # dice loss
        loss_fn = DiceLossEx(normalisation=normalisation_name, exclude_classes=exclude_classes,
                             target_one_hot=target_one_hot)
    elif loss_fun_name == "GDL":
        # general dice loss
        loss_fn = DiceLossEx(normalisation=normalisation_name, weight=weights, exclude_classes=exclude_classes,
                             target_one_hot=target_one_hot)
    elif loss_fun_name == "TL":
        # tversky loss
        loss_fn = TverskyLoss(normalisation=normalisation_name, exclude_classes=exclude_classes,
                              target_one_hot=target_one_hot)
    elif loss_fun_name == "FTL":
        # focal tversky loss
        loss_fn = FocalTverskyLoss(normalisation=normalisation_name, exclude_classes=exclude_classes,
                                   target_one_hot=target_one_hot)
    elif loss_fun_name == "CE":
        # cross entropy loss
        # uses softmax
        if exclude_classes is not None:
            ignore_index = exclude_classes[0]
            loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            loss_fn = nn.CrossEntropyLoss()
    elif loss_fun_name == "WCE":
        # cross entropy loss
        # uses softmax
        if exclude_classes is not None:
            ignore_index = exclude_classes[0]
            loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        assert False, f"no loss function initialised, because {loss_fun_name} unknown."

    return loss_fn


if __name__ == "__main__":
    torch.manual_seed(0)

    # loss fun
    loss_fn_de = DiceLossEx(normalisation="softmax", exclude_classes=[1, 2], target_one_hot=False)
    loss_fn_t = TverskyLoss(normalisation="softmax")
    loss_fn_f = FocalTverskyLoss(normalisation="softmax")
    target = torch.randint(4, (3, 5, 5))
    pred = torch.rand(3, 4, 5, 5)

    loss_de = loss_fn_de(pred, target)
    loss_t = loss_fn_t(pred, target)
    loss_f = loss_fn_f(pred, target)

    print('dice_ex:', loss_de, 'tversky:', loss_t, 'focal tversky:', loss_f)

    target = torch.randint(2, (3, 4, 5, 5))
    pred = torch.rand(3, 4, 5, 5)
    loss_fn_de = DiceLossEx(normalisation="softmax", exclude_classes=[1, 2], target_one_hot=True)

    loss_de = loss_fn_de(pred, target)
    print('dice_ex:', loss_de)

    # confusion matrix
    target = torch.randint(2, (2, 4, 4))
    pred = torch.rand(2, 2, 4, 4)

    print('target', target)
    print('pred', pred)

    tp, fp, tn, fn = confusion_matrix(pred, target, num_classes=2, batch=True, target_one_hot=False)

    print('tp', tp, 'fp', fp, 'tn', tn, 'fn', fn)

    target = torch.randint(2, (2, 2, 4, 4))
    pred = torch.rand(2, 2, 4, 4)

    print('target', target)
    print('pred', pred)

    tp, fp, tn, fn = confusion_matrix(pred, target, num_classes=2, batch=True, target_one_hot=True)

    print('tp', tp, 'fp', fp, 'tn', tn, 'fn', fn)

    i = Identity()
    x = torch.tensor([5, 7])
    print(i(x))

    get_loss_fun(loss_fun_name='DL', normalisation_name='softmax')