import torch


def _binary_focal_loss_from_logits(labels, logits, gamma, pos_weight=None,
                                  label_smoothing=None):
    """Compute focal loss from probabilities.
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's: binary class labels.
    p : tf.Tensor
        Estimated probabilities for the positive class.
    gamma : float
        Focusing parameter.
    pos_weight : float or None
        If not None, losses for the positive class will be scaled by this
        weight.
    label_smoothing : float or None
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels `y_true` are squeezed toward 0.5, with larger values
        of `label_smoothing` leading to label values closer to 0.5.
    Returns
    -------
    tf.Tensor
        The loss for each example.
    """

    p = torch.sigmoid(logits)

    _EPSILON = 1e-9
    # Predicted probabilities for the negative class
    q = 1 - p

    # For numerical stability (so we don't inadvertently take the log of 0)
    p = torch.clamp(p, min=_EPSILON)
    q = torch.clamp(q, min=_EPSILON)

    # Loss for the positive examples
    pos_loss = -(q ** gamma) * torch.log(p)
    if pos_weight is not None:
        pos_loss *= pos_weight

    # Loss for the negative examples
    neg_loss = -(p ** gamma) * torch.log(q)

    # Combine loss terms
    if label_smoothing is None:
        labels = labels.bool()
        loss = torch.where(labels, pos_loss, neg_loss)
    else:
        labels = _process_labels(labels=labels, label_smoothing=label_smoothing,
                                 dtype=p.dtype)
        loss = labels * pos_loss + (1 - labels) * neg_loss

    return loss


def _process_labels(labels, label_smoothing, dtype):
    """Pre-process a binary label tensor, maybe applying smoothing.
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's.
    label_smoothing : float or None
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels `y_true` are squeezed toward 0.5, with larger values
        of `label_smoothing` leading to label values closer to 0.5.
    dtype : tf.dtypes.DType
        Desired type of the elements of `labels`.
    Returns
    -------
    tf.Tensor
        The processed labels.
    """
    # labels = tf.dtypes.cast(labels, dtype=dtype)
    if label_smoothing is not None:
        labels = (1 - label_smoothing) * labels + label_smoothing * 0.5
    return labels
