import torch
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_for_layer_info():
    import sys
    sys.path.append("./mainldm")
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    
    config = OmegaConf.load("./mainldm/models/ldm/celeba256/config.yaml")
    
    print(f"Loading model from ./mainldm/models/ldm/celeba256/model.ckpt")
    pl_sd = torch.load("./mainldm/models/ldm/celeba256/model.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    return model


def split_single_cluster_file(cluster_file_path, output_dir, model):
    logger.info(f"Loading cluster file: {cluster_file_path}")
    cluster_data = torch.load(cluster_file_path)
    
    logger.info(f"Cluster file contents: {cluster_data.keys()}")
    
    if 'clusters' in cluster_data and 'assignments' in cluster_data and 'densities' in cluster_data:
        logger.info("Detected single combined cluster file")
        logger.info(f"Total clusters: {len(cluster_data['clusters'])}")
        logger.info(f"Total assignments: {cluster_data['assignments'].numel()}")

        logger.error("Cannot split a single combined cluster file without layer mapping!")
        logger.error("You need to cluster each layer separately.")
        logger.error("\nPlease use a clustering script that creates per-layer cluster files.")
        return False
    
    return False


def create_per_layer_clusters_hist(model, output_dir, n_clusters=100):

    import numpy as np
    import torch
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Creating per-layer histogram-based cluster files...")
    logger.info(f"Using {n_clusters} histogram bins per layer")
    
    diff_model = model.model.diffusion_model
    layer_count = 0
    
    for name, module in diff_model.named_modules():
        
        if not (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            continue
        if not hasattr(module, "weight"):
            continue
        
        weights = module.weight.data.cpu().numpy().flatten()
        
        if len(weights) < n_clusters:
            logger.warning(f"Skipping {name}: not enough weights ({len(weights)})")
            continue
        
        logger.info(f"\nHistogram clustering layer: {name}")
        logger.info(f"  Weight shape: {module.weight.shape}")
        logger.info(f"  Total weights: {len(weights):,}")

        counts, bin_edges = np.histogram(weights, bins=n_clusters)
        
        cluster_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        clusters = torch.from_numpy(cluster_centers).float()
        
        assignments = np.digitize(weights, bin_edges) - 1
        assignments = np.clip(assignments, 0, n_clusters - 1)
        
        densities = (counts / counts.sum()).astype(np.float32)
        densities = torch.from_numpy(densities)

        assignments = torch.from_numpy(assignments).long()
        assignments = assignments.reshape(module.weight.shape)

        cluster_info = {
            "clusters": clusters,
            "assignments": assignments,
            "densities": densities,
            "layer_shape": module.weight.shape,
            "num_weights": len(weights),
        }
        
        layer_name = f"model.{name}"
        output_file = os.path.join(output_dir, f"{layer_name}_clusters.pth")
        torch.save(cluster_info, output_file)
        
        logger.info(f"  ✓ Saved: {output_file}")
        logger.info(f"  Density range: [{densities.min():.4f}, {densities.max():.4f}]")
        logger.info(f"  Cluster center range: [{clusters.min():.4f}, {clusters.max():.4f}]")
        
        layer_count += 1
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Successfully created histogram cluster files for {layer_count} layers")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'=' * 80}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_file", type=str, default="./clustering_output/model_weights_clusters.pth",
                       help="Path to single cluster file (if splitting)")
    parser.add_argument("--output_dir", type=str, default="./clustering_output_per_layer",
                       help="Output directory for per-layer cluster files")
    parser.add_argument("--mode", type=str, choices=["split", "create"], default="create",
                       help="split: Split existing file (not supported), create: Create from scratch")
    parser.add_argument("--n_clusters", type=int, default=100,
                       help="Number of clusters per layer")
    
    args = parser.parse_args()
    
    if args.mode == "split":
        logger.error("Split mode is not supported - single cluster files cannot be split without layer mapping")
        logger.error("Please use --mode create to create per-layer clusters from scratch")
        exit(1)
    
    logger.info("Loading model...")
    model = load_model_for_layer_info()
    logger.info("Model loaded successfully")
    
    success = create_per_layer_clusters_hist(
        model,
        args.output_dir,
        n_clusters=args.n_clusters,
    )
    
    if success:
        logger.info("\n✓ Done! You can now run main.py with:")
        logger.info(f"  --clustering_dir {args.output_dir}")
    else:
        logger.error("\n✗ Failed to create cluster files")
