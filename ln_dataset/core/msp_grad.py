# ln_dataset/core/msp_grad.py
import torch
import torch.nn.functional as F


def get_msp_gradient(model, image_tensor, target_class=None):
    """
    Computes the gradient of the Maximum Softmax Probability (MSP)
    with respect to the input image.

    Args:
        model: The classifier (frozen).
        image_tensor: Input image (1, C, H, W), normalized.
        target_class: If None, uses the predicted class.

    Returns:
        saliency_map: (H, W) tensor, normalized [0,1].
        gradients: Raw gradients (1, C, H, W) for adversarial steps.
    """
    image_tensor = image_tensor.detach().clone()
    image_tensor.requires_grad = True

    output = model(image_tensor)
    probs = F.softmax(output, dim=1)

    if target_class is None:
        target_class = probs.argmax(dim=1)

    # Get probability of the target class
    msp_score = probs[0, target_class]

    # Backward pass to get gradients w.r.t input
    model.zero_grad()
    msp_score.backward()

    grads = image_tensor.grad.data

    # Calculate magnitude (saliency)
    saliency = grads.abs().max(dim=1)[0][0]  # Take max across channels -> (H, W)

    # Normalize to [0, 1] for masking usage
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency, grads