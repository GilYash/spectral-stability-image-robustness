"""Spectral stability profiles for robust test-time image classification.

Library modules:
    filters          low-pass filter bank and per-image energy ladder
    analysis         the profile maths: C_k, A_k, S_k and softmax fusion (numpy)
    metrics          accuracy, ECE, oracle bound, transfer rates; FusionConfig
    splits           corruption families, leave-one-family-out, disjoint image halves
    cache            build and load the profile cache
    models           frozen ResNet-18 / DeiT-Tiny loaders
    imagenet_labels  image path -> true ImageNet index
"""
