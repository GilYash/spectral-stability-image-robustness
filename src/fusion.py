import torch


def reliability_scores(
    confidence,
    agreement,
    stability,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    """
    Compute one reliability score per spectral view.

    r_k = alpha * C_k + beta * A_k + gamma * S_k
    """

    if not (
        confidence.shape
        == agreement.shape
        == stability.shape
    ):
        raise ValueError(
            "confidence, agreement, and stability "
            "must have the same shape"
        )

    return (
        alpha * confidence
        + beta * agreement
        + gamma * stability
    )


def fusion_weights(scores, tau=1.0):
    """
    Convert reliability scores into normalized fusion weights.

    w_k = softmax(r_k / tau)
    """

    if tau <= 0:
        raise ValueError("tau must be positive")

    return torch.softmax(scores / tau, dim=0)


def fuse_probabilities(probs, weights):
    """
    Weighted average of the class-probability vectors.

    probs   : (num_views, num_classes)
    weights : (num_views,)
    """

    if probs.ndim != 2:
        raise ValueError(
            "probs must have shape "
            "(num_views, num_classes)"
        )

    if weights.ndim != 1:
        raise ValueError(
            "weights must have shape (num_views,)"
        )

    if probs.shape[0] != weights.shape[0]:
        raise ValueError(
            "number of views in probs and weights must match"
        )

    return (weights.unsqueeze(1) * probs).sum(dim=0)


def adaptive_fusion(
    probs,
    confidence,
    agreement,
    stability,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    tau=1.0,
):
    """
    Full adaptive fusion pipeline.

    Returns:
        fused_probs
        weights
        scores
    """

    scores = reliability_scores(
        confidence,
        agreement,
        stability,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    weights = fusion_weights(
        scores,
        tau=tau,
    )

    fused_probs = fuse_probabilities(
        probs,
        weights,
    )

    return fused_probs, weights, scores