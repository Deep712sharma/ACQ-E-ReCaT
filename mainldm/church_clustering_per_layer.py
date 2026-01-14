"""
Automatic detection and fixing of cluster file issues
Run this before main.py to ensure clustering is set up correctly
"""

import torch
import os
import sys
import argparse
import logging
from pathlib import Path

# Fix OpenBLAS threading issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def check_clustering_setup(clustering_dir):
    """
    Check if clustering is set up correctly
    
    Returns:
        status: 'ok', 'single_file', 'missing', or 'empty'
        message: Description of the issue
    """
    if not os.path.exists(clustering_dir):
        return 'missing', f"Directory does not exist: {clustering_dir}"
    
    cluster_files = [f for f in os.listdir(clustering_dir) if f.endswith('_clusters.pth')]
    
    if len(cluster_files) == 0:
        return 'empty', f"No cluster files found in {clustering_dir}"
    
    if len(cluster_files) == 1:
        # Check if it's a combined file
        single_file = os.path.join(clustering_dir, cluster_files[0])
        data = torch.load(single_file)
        
        if 'clusters' in data and 'assignments' in data:
            # This is a single combined file
            file_name = cluster_files[0].replace('_clusters.pth', '')
            if file_name in ['model_weights', 'all_weights', 'combined']:
                return 'single_file', f"Found single combined cluster file: {cluster_files[0]}"
    
    # Check if files have proper naming
    first_file = cluster_files[0]
    if not first_file.startswith('model.'):
        return 'wrong_naming', f"Cluster files don't follow naming convention (should start with 'model.')"
    
    return 'ok', f"Found {len(cluster_files)} properly named cluster files"


def auto_fix(clustering_dir, output_dir, n_clusters=100):
    """
    Automatically fix clustering setup
    """
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
        logger.warning(f"\n⚠️  Directory {clustering_dir} does not exist")
        logger.info(f"Creating per-layer clusters in: {output_dir}")
        return create_per_layer_clusters(output_dir, n_clusters)
    
    if status == 'empty':
        logger.warning(f"\n⚠️  No cluster files found in {clustering_dir}")
        logger.info(f"Creating per-layer clusters in: {output_dir}")
        return create_per_layer_clusters(output_dir, n_clusters)
    
    if status == 'single_file':
        logger.warning(f"\n⚠️  Found single combined cluster file")
        logger.warning("This format is not supported by the entropy model")
        logger.info(f"Creating per-layer clusters in: {output_dir}")
        return create_per_layer_clusters(output_dir, n_clusters)
    
    if status == 'wrong_naming':
        logger.warning(f"\n⚠️  Cluster files have incorrect naming")
        logger.warning("Files should be named: model.layer_name_clusters.pth")
        logger.info(f"Creating new per-layer clusters in: {output_dir}")
        return create_per_layer_clusters(output_dir, n_clusters)
    
    return False


def create_per_layer_clusters(output_dir = "./cluster_layer_church", n_clusters=100):
    """Create per-layer cluster files from scratch"""
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    import gc
    
    logger.info("\n" + "="*80)
    logger.info("CREATING PER-LAYER CLUSTER FILES")
    logger.info("="*80)
    
    # Load model
    logger.info("\nStep 1/3: Loading model...")
    try:
        sys.path.append("./mainldm")
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        
        config = OmegaConf.load("./mainldm/models/ldm/lsun_churches256/config.yaml")
        pl_sd = torch.load("./mainldm/models/ldm/lsun_churches256/model.ckpt", map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        
        # Free up memory
        del pl_sd
        gc.collect()
        
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Cluster each layer
    logger.info(f"\nStep 2/3: Clustering layers (this may take 30-60 minutes)...")
    logger.info(f"Using {n_clusters} clusters per layer")
    
    diff_model = model.model.diffusion_model
    layer_count = 0
    skipped_count = 0
    
    for name, module in diff_model.named_modules():
        # Only cluster conv and linear layers
        if not (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            continue
        
        if not hasattr(module, 'weight'):
            continue
        
        try:
            # Get weights and convert to numpy
            weights = module.weight.data.cpu().numpy().flatten()
            
            if len(weights) < n_clusters * 10:
                skipped_count += 1
                continue
            
            if layer_count % 10 == 0:
                logger.info(f"  Processing layer {layer_count + 1}... ({name})")
            
            # Sample weights - use smaller sample size to reduce memory
            n_samples = min(len(weights), min(50000, max(10000, len(weights) // 10)))
            indices = np.random.choice(len(weights), n_samples, replace=False)
            sampled_weights = weights[indices].reshape(-1, 1)
            
            # Cluster with explicit n_init and smaller batch size
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=512,  # Reduced from 1024
                max_iter=100,
                n_init=3,  # Explicitly set to suppress warning
                verbose=0
            )
            kmeans.fit(sampled_weights)
            
            # Get assignments for all weights in batches to avoid memory issues
            batch_size = 100000
            assignments_list = []
            
            for i in range(0, len(weights), batch_size):
                batch = weights[i:i+batch_size].reshape(-1, 1)
                batch_assignments = kmeans.predict(batch)
                assignments_list.append(batch_assignments)
                
            assignments = np.concatenate(assignments_list)
            assignments = torch.from_numpy(assignments).long()
            
            # Calculate densities
            unique, counts = np.unique(assignments.numpy(), return_counts=True)
            densities = np.zeros(n_clusters)
            for cluster_id, count in zip(unique, counts):
                densities[cluster_id] = count / len(assignments)
            
            # Save
            cluster_info = {
                'clusters': torch.from_numpy(kmeans.cluster_centers_.flatten()).float(),
                'assignments': assignments.reshape(module.weight.shape),
                'densities': torch.from_numpy(densities).float(),
                'layer_shape': module.weight.shape,
                'num_weights': len(weights)
            }
            
            layer_name = f"model.{name}"
            output_file = os.path.join(output_dir, f"{layer_name}_clusters.pth")
            torch.save(cluster_info, output_file)
            
            layer_count += 1
            
            # Clean up memory after each layer
            del weights, sampled_weights, assignments, cluster_info
            gc.collect()
            
        except Exception as e:
            logger.warning(f"  Failed to cluster layer {name}: {e}")
            skipped_count += 1
            continue
    
    logger.info(f"\n✓ Successfully clustered {layer_count} layers")
    if skipped_count > 0:
        logger.info(f"  (Skipped {skipped_count} layers with too few weights or errors)")
    
    logger.info(f"\nStep 3/3: Verification...")
    verify_files = [f for f in os.listdir(output_dir) if f.endswith('_clusters.pth')]
    logger.info(f"✓ Created {len(verify_files)} cluster files in: {output_dir}")
    
    logger.info("\n" + "="*80)
    logger.info("SETUP COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nYou can now run main.py with:")
    logger.info(f"  --clustering_dir {output_dir}")
    logger.info("\nExample:")
    logger.info(f"  python main.py --use_cluster_quant --use_entropy_model \\")
    logger.info(f"    --clustering_dir {output_dir}")
    logger.info("="*80)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatic detection and fixing of cluster file issues"
    )
    parser.add_argument("--clustering_dir", type=str, default="./clustering_output",
                       help="Directory to check for cluster files")
    parser.add_argument("--output_dir", type=str, default="./clustering_per_layer",
                       help="Output directory for fixed cluster files")
    parser.add_argument("--n_clusters", type=int, default=100,
                       help="Number of clusters per layer")
    parser.add_argument("--check_only", action="store_true",
                       help="Only check status, don't fix")
    
    args = parser.parse_args()
    
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
        success = auto_fix(args.clustering_dir, args.output_dir, args.n_clusters)
        sys.exit(0 if success else 1)