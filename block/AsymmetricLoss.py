# Auto-generated single-file for AsymmetricLoss
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- mmpretrain.models.losses.utils.convert_to_one_hot ----
def convert_to_one_hot(targets: torch.Tensor, classes) -> torch.Tensor:
    """This function converts target class indices to one-hot vectors, given
    the number of classes.

    Args:
        targets (Tensor): The ground truth label of the prediction
                with shape (N, 1)
        classes (int): the number of classes.

    Returns:
        Tensor: Processed loss values.
    """
    assert (torch.max(targets).item() <
            classes), 'Class Index must be less than number of classes'
    one_hot_targets = F.one_hot(
        targets.long().squeeze(-1), num_classes=classes)
    return one_hot_targets

# ---- mmpretrain.models.losses.utils.reduce_loss ----
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

# ---- mmpretrain.models.losses.utils.weight_reduce_loss ----
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

# ---- mmpretrain.models.losses.asymmetric_loss.asymmetric_loss ----
def asymmetric_loss(pred,
                    target,
                    weight=None,
                    gamma_pos=1.0,
                    gamma_neg=4.0,
                    clip=0.05,
                    reduction='mean',
                    avg_factor=None,
                    use_sigmoid=True,
                    eps=1e-8):
    r"""asymmetric loss.

    Please refer to the `paper <https://arxiv.org/abs/2009.14119>`__ for
    details.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Defaults to None.
        gamma_pos (float): positive focusing parameter. Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We usually set
            gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        use_sigmoid (bool): Whether the prediction uses sigmoid instead
            of softmax. Defaults to True.
        eps (float): The minimum value of the argument of logarithm. Defaults
            to 1e-8.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    if use_sigmoid:
        pred_sigmoid = pred.sigmoid()
    else:
        pred_sigmoid = nn.functional.softmax(pred, dim=-1)

    target = target.type_as(pred)

    if clip and clip > 0:
        pt = (1 - pred_sigmoid +
              clip).clamp(max=1) * (1 - target) + pred_sigmoid * target
    else:
        pt = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    asymmetric_weight = (1 - pt).pow(gamma_pos * target + gamma_neg *
                                     (1 - target))
    loss = -torch.log(pt.clamp(min=eps)) * asymmetric_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

# ---- AsymmetricLoss (target) ----
class AsymmetricLoss(nn.Module):
    """asymmetric loss.

    Args:
        gamma_pos (float): positive focusing parameter.
            Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We
            usually set gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss into
            a scalar.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        use_sigmoid (bool): Whether the prediction uses sigmoid instead
            of softmax. Defaults to True.
        eps (float): The minimum value of the argument of logarithm. Defaults
            to 1e-8.
    """

    def __init__(self,
                 gamma_pos=0.0,
                 gamma_neg=4.0,
                 clip=0.05,
                 reduction='mean',
                 loss_weight=1.0,
                 use_sigmoid=True,
                 eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        r"""asymmetric loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*), N or (N,1).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1):
            target = convert_to_one_hot(target.view(-1, 1), pred.shape[-1])
        loss_cls = self.loss_weight * asymmetric_loss(
            pred,
            target,
            weight,
            gamma_pos=self.gamma_pos,
            gamma_neg=self.gamma_neg,
            clip=self.clip,
            reduction=reduction,
            avg_factor=avg_factor,
            use_sigmoid=self.use_sigmoid,
            eps=self.eps)
        return loss_cls
