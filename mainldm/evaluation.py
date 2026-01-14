import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader, Dataset
import clip  # pip install git+https://github.com/openai/CLIP.git
from scipy import linalg


class ImageFolderWithCaptions(Dataset):
    def __init__(self, img_dir, captions_file=None, image_size=256):
        self.img_dir = img_dir
        self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.lower().endswith(('.png', '.jpg', '.webp', '.jpeg'))])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.transform = self.clip_preprocess
        self.captions = None
        if captions_file and os.path.exists(captions_file):
            with open(captions_file, 'r', encoding='utf-8') as f:
                self.captions = [line.strip() for line in f.readlines()]
            assert len(self.captions) == len(self.image_paths), \
                f"Number of captions ({len(self.captions)}) must match number of images ({len(self.image_paths)})."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption = self.captions[idx] if self.captions else ""
        return image, caption, img_path


class ImageFolder(Dataset):
    """Simple image folder loader for FID and Inception Score"""
    def __init__(self, img_dir, image_size=256):
        self.img_dir = img_dir
        self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.lower().endswith(('.png', '.jpg', '.webp', '.jpeg'))])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image


# -----------------------------
# Inception Feature Extractor for FID
# -----------------------------
class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()

    def forward(self, x):
        # 299x299 expected
        features = self.model(x)
        return features


def calculate_activation_statistics(dataloader, model, device):
    model.eval()
    activations = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)

            if images.shape[1] != 3:
                images = images.repeat(1, 3, 1, 1)

            pred = model(images)
            activations.append(pred.cpu().numpy())

    if not activations:
        raise ValueError("No images found in the dataset directory. Please check that the directory contains valid image files (.png, .jpg, .jpeg).")

    activations = np.concatenate(activations, axis=0)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("FID calculation: Adding epsilon to covariance matrices for stability")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def compute_inception_score(dataloader, device, splits=10):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Inception Score"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)

            if images.shape[1] != 3:
                images = images.repeat(1, 3, 1, 1)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    n = preds.shape[0]
    split_scores = []
    split_size = n // splits

    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size, :]
        py = np.mean(part, axis=0)
        scores = []
        for j in range(part.shape[0]):
            pyx = part[j, :]
            kl = pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))
            scores.append(np.sum(kl))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def compute_clip_score(dataloader, device):
    """
    Compute CLIP Score between images and their captions.
    Higher scores indicate better image-text alignment.
    """
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing CLIP Score"):
            images, captions, _ = batch
            images = images.to(device)
            
            # Tokenize captions
            text_tokens = clip.tokenize(captions, truncate=True).to(device)
            
            # Get embeddings
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (image_features * text_features).sum(dim=-1)
            scores.extend(similarity.cpu().numpy())
    
    scores = np.array(scores)
    return np.mean(scores) * 100, np.std(scores) * 100  # Scale to 0-100 range


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute FID, Inception Score, and CLIP Score for generated images")
    parser.add_argument("--real_dir", type=str, help="Directory of real images (required for FID)")
    parser.add_argument("--fake_dir", type=str, required=True, help="Directory of generated images")
    parser.add_argument("--captions_file", type=str, default=None, 
                        help="Text file with captions (one per line, matching image order)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splits", type=int, default=10, help="Number of splits for Inception Score")
    parser.add_argument("--only_clip", action="store_true", help="Compute only CLIP Score (requires --captions_file)")
    args = parser.parse_args()

    if args.only_clip and not args.captions_file:
        parser.error("--only_clip requires --captions_file")
    if not args.only_clip and not args.real_dir:
        parser.error("--real_dir is required unless --only_clip is set")

    device = args.device
    print(f"Using device: {device}\n")

    if not args.only_clip:
        # ============================================
        # 1. Compute FID
        # ============================================
        print("=" * 50)
        print("Computing FID (Frechet Inception Distance)...")
        print("=" * 50)
        fid_model = InceptionFeatureExtractor().to(device)
        real_dataset = ImageFolder(args.real_dir, image_size=299)
        fake_dataset = ImageFolder(args.fake_dir, image_size=299)
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)
        fake_loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False)

        mu_real, sigma_real = calculate_activation_statistics(real_loader, fid_model, device)
        mu_fake, sigma_fake = calculate_activation_statistics(fake_loader, fid_model, device)
        fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        print(f"✓ FID: {fid_value:.2f}\n")

        # ============================================
        # 2. Compute Inception Score
        # ============================================
        print("=" * 50)
        print("Computing Inception Score (IS)...")
        print("=" * 50)
        fake_loader_is = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False)
        is_mean, is_std = compute_inception_score(fake_loader_is, device, splits=args.splits)
        print(f"✓ Inception Score: {is_mean:.2f} ± {is_std:.2f}\n")
    else:
        fake_dataset = ImageFolder(args.fake_dir, image_size=256)  # For CLIP, use 256

    # ============================================
    # 3. Compute CLIP Score
    # ============================================
    if args.captions_file:
        print("=" * 50)
        print("Computing CLIP Score...")
        print("=" * 50)
        clip_dataset = ImageFolderWithCaptions(args.fake_dir, captions_file=args.captions_file)
        clip_loader = DataLoader(clip_dataset, batch_size=args.batch_size, shuffle=False)
        
        clip_mean, clip_std = compute_clip_score(clip_loader, device)
        print(f"✓ CLIP Score: {clip_mean:.2f} ± {clip_std:.2f}\n")
    elif not args.only_clip:
        print("=" * 50)
        print("CLIP Score: SKIPPED (no captions file provided)")
        print("=" * 50)
        print("To compute CLIP Score, provide --captions_file argument")
        print("Format: one caption per line, matching image order\n")

    # ============================================
    # Summary
    # ============================================
    print("=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)
    if not args.only_clip:
        print(f"FID ↓:              {fid_value:.2f}")
        print(f"Inception Score ↑:  {is_mean:.2f} ± {is_std:.2f}")
    if args.captions_file:
        print(f"CLIP Score ↑:       {clip_mean:.2f} ± {clip_std:.2f}")
    print("=" * 50)
    print("\n↓ = Lower is better | ↑ = Higher is better")