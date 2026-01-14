"""
Script to split a single cluster file containing all weights 
into per-layer cluster files that can be used by the entropy model
"""

import torch
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_for_layer_info():
    """Load model to get layer information"""
    import sys
    sys.path.append("./mainldm")
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    
    config = OmegaConf.load("./mainldm/models/ldm/lsun_beds256/config.yaml")

    print(f"Loading model from ./mainldm/models/ldm/lsun_beds256/model.ckpt")
    pl_sd = torch.load("./mainldm/models/ldm/lsun_beds256/model.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    return model


def split_single_cluster_file(cluster_file_path, output_dir, model):
    """
    Split a single cluster file into per-layer files
    
    Args:
        cluster_file_path: Path to single cluster file (e.g., model_weights_clusters.pth)
        output_dir: Directory to save per-layer cluster files
        model: Model to get layer information from
    """
    
    logger.info(f"Loading cluster file: {cluster_file_path}")
    cluster_data = torch.load(cluster_file_path)
    
    logger.info(f"Cluster file contents: {cluster_data.keys()}")
    
    # Check if this is a single combined file
    if 'clusters' in cluster_data and 'assignments' in cluster_data and 'densities' in cluster_data:
        logger.info("Detected single combined cluster file")
        logger.info(f"Total clusters: {len(cluster_data['clusters'])}")
        logger.info(f"Total assignments: {cluster_data['assignments'].numel()}")
        
        # This approach won't work well because we need to know which 
        # assignments belong to which layer
        logger.error("Cannot split a single combined cluster file without layer mapping!")
        logger.error("You need to cluster each layer separately.")
        logger.error("\nPlease use a clustering script that creates per-layer cluster files.")
        return False
    
    return False


def create_per_layer_clusters(model, output_dir, n_clusters=100, sample_ratio=0.1):
    """
    Create cluster files for each layer from scratch
    
    Args:
        model: Model to cluster
        output_dir: Output directory
        n_clusters: Number of clusters per layer
        sample_ratio: Ratio of weights to sample for clustering
    """
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Creating per-layer cluster files...")
    logger.info(f"Using {n_clusters} clusters per layer")
    
    # Get diffusion model
    diff_model = model.model.diffusion_model
    
    layer_count = 0
    
    for name, module in diff_model.named_modules():
        # Only cluster conv and linear layers
        if not (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            continue
        
        if not hasattr(module, 'weight'):
            continue
        
        weights = module.weight.data.cpu().numpy().flatten()
        
        if len(weights) < n_clusters * 10:
            logger.warning(f"Skipping {name}: too few weights ({len(weights)})")
            continue
        
        logger.info(f"\nClustering layer: {name}")
        logger.info(f"  Weight shape: {module.weight.shape}")
        logger.info(f"  Total weights: {len(weights):,}")
        
        # Sample weights for clustering
        n_samples = min(len(weights), int(len(weights) * sample_ratio))
        n_samples = max(n_samples, n_clusters * 10)  # At least 10x clusters
        
        indices = np.random.choice(len(weights), n_samples, replace=False)
        sampled_weights = weights[indices].reshape(-1, 1)
        
        logger.info(f"  Sampling {n_samples:,} weights for clustering")
        
        # Perform clustering
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=1024,
            max_iter=100
        )
        
        kmeans.fit(sampled_weights)
        
        # Get cluster centers
        clusters = torch.from_numpy(kmeans.cluster_centers_.flatten()).float()
        
        # Assign all weights to clusters
        all_weights = weights.reshape(-1, 1)
        assignments = kmeans.predict(all_weights)
        assignments = torch.from_numpy(assignments).long()
        
        # Calculate densities (proportion of weights in each cluster)
        unique, counts = np.unique(assignments.numpy(), return_counts=True)
        densities = np.zeros(n_clusters)
        for cluster_id, count in zip(unique, counts):
            densities[cluster_id] = count / len(assignments)
        densities = torch.from_numpy(densities).float()
        
        # Reshape assignments to match weight shape
        assignments = assignments.reshape(module.weight.shape)
        
        # Save cluster file
        cluster_info = {
            'clusters': clusters,
            'assignments': assignments,
            'densities': densities,
            'layer_shape': module.weight.shape,
            'num_weights': len(weights)
        }
        
        # Use model. prefix for naming consistency
        layer_name = f"model.{name}"
        output_file = os.path.join(output_dir, f"{layer_name}_clusters.pth")
        torch.save(cluster_info, output_file)
        
        logger.info(f"  ✓ Saved: {output_file}")
        logger.info(f"  Clusters: {n_clusters}, Density range: [{densities.min():.4f}, {densities.max():.4f}]")
        
        layer_count += 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Successfully created cluster files for {layer_count} layers")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*80}")
    
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
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                       help="Ratio of weights to sample for clustering")
    
    args = parser.parse_args()
    
    if args.mode == "split":
        logger.error("Split mode is not supported - single cluster files cannot be split without layer mapping")
        logger.error("Please use --mode create to create per-layer clusters from scratch")
        exit(1)
    
    # Load model
    logger.info("Loading model...")
    model = load_model_for_layer_info()
    logger.info("Model loaded successfully")
    
    # Create per-layer clusters
    success = create_per_layer_clusters(
        model,
        args.output_dir,
        n_clusters=args.n_clusters,
        sample_ratio=args.sample_ratio
    )
    
    if success:
        logger.info("\n✓ Done! You can now run main.py with:")
        logger.info(f"  --clustering_dir {args.output_dir}")
    else:
        logger.error("\n✗ Failed to create cluster files")