import torch
import os
import sys
import argparse
import logging
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_model_for_layer_info():
    """Load the LDM model for layer information extraction."""
    sys.path.append("./mainldm")
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    
    config = OmegaConf.load("./mainldm/models/ldm/celeba256/config.yaml")
    
    logger.info(f"Loading model from ./mainldm/models/ldm/celeba256/model.ckpt")
    pl_sd = torch.load("./mainldm/models/ldm/celeba256/model.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    return model


def split_single_cluster_file(cluster_file_path, output_dir, model):
    """Attempt to split a single cluster file (not supported - will show error)."""
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
    """Create histogram-based cluster files for each layer."""
    import numpy as np
    
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


def check_clustering_setup(clustering_dir):
    """Check the status of clustering setup in a directory."""
    if not os.path.exists(clustering_dir):
        return 'missing', f"Directory does not exist: {clustering_dir}"
    
    cluster_files = [f for f in os.listdir(clustering_dir) if f.endswith('_clusters.pth')]
    
    if len(cluster_files) == 0:
        return 'empty', f"No cluster files found in {clustering_dir}"
    
    if len(cluster_files) == 1:
        single_file = os.path.join(clustering_dir, cluster_files[0])
        data = torch.load(single_file)
        
        if 'clusters' in data and 'assignments' in data:
            file_name = cluster_files[0].replace('_clusters.pth', '')
            if file_name in ['model_weights', 'all_weights', 'combined']:
                return 'single_file', f"Found single combined cluster file: {cluster_files[0]}"
    
    first_file = cluster_files[0]
    if not first_file.startswith('model.'):
        return 'wrong_naming', f"Cluster files don't follow naming convention (should start with 'model.')"
    
    return 'ok', f"Found {len(cluster_files)} properly named cluster files"


def auto_fix(model, clustering_dir, output_dir, n_clusters=100):
    """Automatically detect and fix clustering issues."""
    logger.info("="*80)
    logger.info("AUTOMATIC CLUSTER SETUP FIX")
    logger.info("="*80)
    
    status, message = check_clustering_setup(clustering_dir)
    
    logger.info(f"\nStatus: {status}")
    logger.info(f"Message: {message}")
    
    if status == 'ok':
        logger.info("\n✓ Clustering is already set up correctly!")
        logger.info(f"✓ You can use: --clustering_dir {clustering_dir}")
        return True
    
    if status == 'missing':
        logger.warning(f"\n⚠ Directory {clustering_dir} does not exist")
        logger.info(f"Creating per-layer clusters in: {output_dir}")
        return create_per_layer_clusters_hist(model, output_dir, n_clusters)
    
    if status == 'empty':
        logger.warning(f"\n⚠ No cluster files found in {clustering_dir}")
        logger.info(f"Creating per-layer clusters in: {output_dir}")
        return create_per_layer_clusters_hist(model, output_dir, n_clusters)
    
    if status == 'single_file':
        logger.warning(f"\n⚠ Found single combined cluster file")
        logger.warning("This format is not supported by the entropy model")
        logger.info(f"Creating per-layer clusters in: {output_dir}")
        return create_per_layer_clusters_hist(model, output_dir, n_clusters)
    
    if status == 'wrong_naming':
        logger.warning(f"\n⚠ Cluster files have incorrect naming")
        logger.warning("Files should be named: model.layer_name_clusters.pth")
        logger.info(f"Creating new per-layer clusters in: {output_dir}")
        return create_per_layer_clusters_hist(model, output_dir, n_clusters)
    
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster management: create, split, or auto-fix cluster files"
    )
    parser.add_argument("--cluster_file", type=str, default="./clustering_output/model_weights_clusters.pth",
                       help="Path to single cluster file (if splitting)")
    parser.add_argument("--clustering_dir", type=str, default="./clustering_output",
                       help="Directory to check for cluster files")
    parser.add_argument("--output_dir", type=str, default="./clustering_per_layer",
                       help="Output directory for per-layer cluster files")
    parser.add_argument("--mode", type=str, choices=["split", "create", "auto"], default="auto",
                       help="split: Split existing file (not supported), create: Create from scratch, auto: Auto-detect and fix")
    parser.add_argument("--n_clusters", type=int, default=100,
                       help="Number of clusters per layer")
    parser.add_argument("--check_only", action="store_true",
                       help="Only check status, don't fix (only for auto mode)")
    
    args = parser.parse_args()
    
    logger.info("Loading model...")
    model = load_model_for_layer_info()
    logger.info("Model loaded successfully\n")
    
    if args.mode == "split":
        logger.error("Split mode is not supported - single cluster files cannot be split without layer mapping")
        logger.error("Please use --mode create to create per-layer clusters from scratch")
        logger.error("Or use --mode auto to automatically detect and fix issues")
        sys.exit(1)
    
    elif args.mode == "create":
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
        
        sys.exit(0 if success else 1)
    
    elif args.mode == "auto":
        if args.check_only:
            status, message = check_clustering_setup(args.clustering_dir)
            print(f"\nStatus: {status}")
            print(f"Message: {message}")
            
            if status == 'ok':
                print("\n✓ Everything is set up correctly!")
                sys.exit(0)
            else:
                print("\n✗ Issues detected. Run without --check_only to fix.")
                sys.exit(1)
        else:
            success = auto_fix(model, args.clustering_dir, args.output_dir, args.n_clusters)
            
            if success:
                logger.info("\n✓ Done! You can now run main.py with:")
                logger.info(f"  --clustering_dir {args.output_dir}")
            
            sys.exit(0 if success else 1)