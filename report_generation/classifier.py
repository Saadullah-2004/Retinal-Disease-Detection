#!/usr/bin/env python


import os, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import ViTForImageClassification, ViTFeatureExtractor

# allow‑list ViTConfig so torch.load(weights_only=True) can un‑pickle it
from transformers.models.vit.configuration_vit import ViTConfig
import torch.serialization as _ts
_ts.add_safe_globals([ViTConfig])

class RetinalDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.data_frame.iloc[idx]["id_code"] + ".png")
        image = Image.open(img_path).convert("RGB")
        diagnosis = self.data_frame.iloc[idx]["diagnosis"]
        pixel_values = self.extractor(images=image,
                                      return_tensors="pt")["pixel_values"]
        return pixel_values.squeeze(), torch.tensor(diagnosis, dtype=torch.long)

class RetinalClassifier:
    def __init__(self, num_classes: int = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        self.class_names = [
            "No DR (0)", "Mild NPDR (1)", "Moderate NPDR (2)",
            "Severe NPDR (3)", "Proliferative DR (4)"
        ]

    def train(self, train_loader, val_loader=None,
              num_epochs=10, learning_rate=2e-5):
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        best_acc = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = correct = total = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x).logits
                loss = criterion(logits, y)
                loss.backward(); opt.step()
                running_loss += loss.item()
                total += y.size(0)
                correct += (logits.argmax(1) == y).sum().item()

            print(f"Epoch {epoch+1:2d}  "
                  f"loss {running_loss/len(train_loader):.4f}  "
                  f"acc {100*correct/total:.2f}%")

            if val_loader:
                val_acc, val_loss = self.evaluate(val_loader)
                print(f"          val_loss {val_loss:.4f}  val_acc {val_acc:.2f}%")
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "best_acc": best_acc,
                        "model_config": self.model.config,
                    }, "best_model.pth")
                    print("          ✔ new best model saved")

    def evaluate(self, loader):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x).logits
                loss += criterion(logits, y).item()
                total += y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        return 100*correct/total, loss/len(loader)

    
    def predict_single_image(self, image_path: str):
        self.model.eval()
        img = Image.open(image_path).convert("RGB")
        pixel_values = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224")(images=img,
                                           return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(pixel_values).logits, dim=1)
        conf, pred = probs.max(1)
        return {
            "class_id": pred.item(),
            "class_name": self.class_names[pred.item()],
            "confidence": conf.item(),
            "all_probabilities": probs[0].cpu().numpy()
        }

    def load_model(self, path: str):
        """
        Load a ViT checkpoint that may include a 1 000‑class head.
        Returns the stored accuracy if found, else None.
        """
        ckpt = torch.load(path, map_location=self.device,
                          weights_only=True)  # safe mode

        # unwrap common wrapper
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        pruned = {k: v for k, v in state_dict.items()
                  if not k.startswith("classifier.")}

        self.model.load_state_dict(pruned, strict=False)
        print(f"✔ backbone weights loaded from {path}")

        # return stored accuracy if present
        for key in ("best_acc", "accuracy", "val_accuracy"):
            if isinstance(ckpt, dict) and key in ckpt:
                return ckpt[key] * 100 if ckpt[key] <= 1 else ckpt[key]
        return None

    
    def evaluate_and_print(self, loader):
        from sklearn.metrics import classification_report
        acc, _ = self.evaluate(loader)
        print(f"\nTest Accuracy: {acc:.2f}%\n")
        preds, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                preds.extend(self.model(x).logits.argmax(1).cpu().tolist())
                labels.extend(y.tolist())
        print("Detailed Classification Report:")
        print(classification_report(labels, preds, target_names=self.class_names))

def generate_gradcam(model, image_path, target_layer_name="classifier"):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    device = model.device
    model.model.eval()

    img = Image.open(image_path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    inp = t(img).unsqueeze(0).to(device)

    target_layer = (getattr(model.model, target_layer_name)
                    if hasattr(model.model, target_layer_name)
                    else model.model.vit.encoder.layer[-1])

    cam = GradCAM(model=model.model, target_layers=[target_layer])
    pred = model.model(inp).logits.argmax(1).item()
    grayscale = cam(inp, targets=[ClassifierOutputTarget(pred)])[0]

    rgb = inp.squeeze().permute(1, 2, 0).cpu().numpy()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    vis = show_cam_on_image(rgb, grayscale, use_rgb=True)
    return Image.fromarray(vis.astype("uint8"))

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Retinal ViT classifier")
    p.add_argument("--mode", required=True, choices=["train", "predict"])
    p.add_argument("--csv_file")
    p.add_argument("--img_dir")
    p.add_argument("--model_path")
    p.add_argument("--image_path")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    if args.mode == "train":
        assert args.csv_file and args.img_dir
        train_loader, val_loader = RetinalDataset(args.csv_file,
                                                  args.img_dir).split = \
            create_data_loaders(args.csv_file, args.img_dir, args.batch_size)
        clf = RetinalClassifier()
        clf.train(train_loader, val_loader, args.epochs)

    elif args.mode == "predict":
        assert args.model_path and args.image_path
        clf = RetinalClassifier()
        stored_acc = clf.load_model(args.model_path)

        res = clf.predict_single_image(args.image_path)
        print(f"\nSingle‑image prediction : {res['class_name']}")
        print(f"Confidence               : {res['confidence']:.4f}")
        if stored_acc is not None:
            print(f"Model accuracy (stored)  : {stored_acc:.2f}%")

        if args.csv_file and args.img_dir:
            test_loader, _ = create_data_loaders(args.csv_file, args.img_dir,
                                                 args.batch_size, train_ratio=0.0)
            clf.evaluate_and_print(test_loader)

        try:
            heatmap = generate_gradcam(clf, args.image_path)
            out = args.image_path.rsplit('.', 1)[0] + "_heatmap.png"
            heatmap.save(out)
            print(f"Heat‑map saved to {out}")
        except ImportError:
            print("Install pytorch_grad_cam for Grad‑CAM visualisation.")
