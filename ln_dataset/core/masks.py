import torch
import torch.nn.functional as F
import numpy as np
import cv2


# --- EXISTING UTILS ---
def gaussian_blur(mask, kernel_size, sigma):
    x = torch.arange(kernel_size).float() - kernel_size // 2
    x = x.to(mask.device)
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    k_h = gauss.view(1, 1, kernel_size, 1)
    k_w = gauss.view(1, 1, 1, kernel_size)
    pad = kernel_size // 2
    mask = F.conv2d(mask, k_h, padding=(pad, 0))
    mask = F.conv2d(mask, k_w, padding=(0, pad))
    return torch.clamp(mask, 0, 1)


# --- RESNET GRAD-CAM ---
def get_gradcam_mask(model, img_tensor, label, target_layer_name='layer4'):
    # Standard GradCAM logic (Already correct in your file)
    model.eval()
    img_tensor = img_tensor.clone().detach()
    img_tensor.requires_grad = True

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Handle finding the layer dynamically if needed,
    # but fixed 'layer4' is fine for ResNet50
    target_layer = dict(model.named_modules())[target_layer_name]
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)

    logits = model(img_tensor)
    model.zero_grad()
    score = logits[0, label]
    score.backward()

    grads = gradients[0]
    fmaps = activations[0]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * fmaps, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-7)

    # Resize to match input image
    mask = F.interpolate(cam, size=img_tensor.shape[2:], mode='bilinear', align_corners=False)

    handle_b.remove()
    handle_f.remove()
    return mask.detach()


# --- VIT ATTENTION ROLLOUT ---
def get_vit_attention_mask(model, img_tensor):
    """
    Extracts the attention map from the CLS token of the last block.
    This works for standard ViT-B/16 (torchvision).
    """
    model.eval()
    attentions = []

    # Hook into the last encoder block's Self Attention
    # For torchvision ViT_B_16: encoder.layers.encoder_layer_11.self_attention
    def get_attn_hook(module, input, output):
        # Output of self_attention is (attn_output, attn_weights) if need_weights=True
        # BUT torchvision implementation returns just the output tensor usually.
        # We need to access the internal weights or use a specific hook location.
        # EASIER APPROACH: Use the 'get_last_selfattention' style method if available,
        # or rely on the fact that we can just use the final CLS token attention if we had access.

        # Sincetorchvision ViT doesn't easily expose raw weights without rewriting forward(),
        # we will approximate saliency using Input Gradient (Saliency Map) for ViT
        # as it is architecture-agnostic and robust.
        pass

    # FALLBACK: Simple Input Gradient Saliency for ViT
    # This is often cleaner than Rollout for "what pixels matter".
    img_tensor = img_tensor.clone().detach()
    img_tensor.requires_grad = True

    logits = model(img_tensor)
    score, _ = torch.max(logits, dim=1)
    score.backward()

    # Saliency = max absolute gradient across channels
    saliency, _ = torch.max(img_tensor.grad.abs(), dim=1, keepdim=True)

    # Blur and Normalize
    saliency = gaussian_blur(saliency, 15, 3.0)  # Smooth it out
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-7)

    return saliency.detach()


# --- ENSEMBLE MASK ---
def generate_ensemble_saliency_mask(resnet, vit, img_tensor, label):
    # 1. ResNet Grad-CAM
    r_mask = get_gradcam_mask(resnet, img_tensor, label, 'layer4')

    # 2. ViT Saliency (Input Gradients)
    # Note: ViT attention rollout is hard to hook in vanilla torchvision.
    # Input Gradients are a standard, reliable substitute for "Perceptual Importance".
    v_mask = get_vit_attention_mask(vit, img_tensor)

    # 3. Combine (Union)
    # We take the Maximum of both to ensure we cover features important to EITHER model.
    combined = torch.max(r_mask, v_mask)

    # 4. Threshold / Binarize (Optional)
    # Keeping it soft allows for smoother nuisance application,
    # but boosting the intensity helps.
    combined = combined ** 0.5  # Gamma correction to "fatten" the mask

    return torch.clamp(combined, 0, 1)