import torch
from bench.detectors import get_detector
from bench.backbones import load_backbone_from_ln_config
from bench.loader import ConfigurableDataset
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":
    device = torch.device("cuda")

    detectors_to_check = ["dice", "odin"]
    backbones_to_check = ["resnet50", "convnext_t", "swin_t"]

    # Manual loader with num_workers=0 to avoid Windows spawn issue
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm
    ])

    val_dataset = ConfigurableDataset("ImageNet-Val", transform)
    train_dataset = ConfigurableDataset("ImageNet-Train", transform)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)

    batch = next(iter(val_loader))
    img = batch["data"].to(device)

    for bb in backbones_to_check:
        print(f"\n{'=' * 60}\n{bb}\n{'=' * 60}")
        model = load_backbone_from_ln_config(bb, device, "ln_dataset/configs/imagenet.yaml")

        for det_name in detectors_to_check:
            detector = get_detector(det_name, "ImageNet-Val")

            # Setup
            try:
                detector.setup(model, {"train": train_loader, "val": val_loader}, None)
            except Exception as e:
                print(f"  {det_name}: SETUP FAILED - {e}")
                continue

            # Inference
            try:
                model.eval()
                if det_name in ["odin", "gradnorm"]:
                    model.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        preds, confs = detector.postprocess(model, img)
                else:
                    with torch.no_grad():
                        preds, confs = detector.postprocess(model, img)

                confs_np = confs.cpu().numpy() if torch.is_tensor(confs) else confs
                print(
                    f"  {det_name:8s}: min={confs_np.min():.4f}, max={confs_np.max():.4f}, mean={confs_np.mean():.4f}")
            except Exception as e:
                print(f"  {det_name}: INFERENCE FAILED - {e}")