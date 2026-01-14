import torch
import os
import sys
import argparse
import logging
from pathlib import Path
from celeb_split_clusters import create_per_layer_clusters_hist, load_model_for_layer_info

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def check_clustering_setup(clustering_dir):

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


def auto_fix(clustering_dir, output_dir, n_clusters=100):

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
        logger.warning(f"\n Directory {clustering_dir} does not exist")
        logger.info(f"Creating per-layer clusters in: {output_dir}")
        return create_per_layer_clusters_hist(model, output_dir, n_clusters)
    
    if status == 'empty':
        logger.warning(f"\n No cluster files found in {clustering_dir}")
        logger.info(f"Creating per-layer clusters in: {output_dir}")
        return create_per_layer_clusters_hist(model, output_dir, n_clusters)
    
    if status == 'single_file':
        logger.warning(f"\n Found single combined cluster file")
        logger.warning("This format is not supported by the entropy model")
        logger.info(f"Creating per-layer clusters in: {output_dir}")
        return create_per_layer_clusters_hist(model, output_dir, n_clusters)
    
    if status == 'wrong_naming':
        logger.warning(f"\n Cluster files have incorrect naming")
        logger.warning("Files should be named: model.layer_name_clusters.pth")
        logger.info(f"Creating new per-layer clusters in: {output_dir}")
        return create_per_layer_clusters_hist(model, output_dir, n_clusters)
    
    return False


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

    logger.info("Loading model...")
    model = load_model_for_layer_info()
    logger.info("Model loaded successfully")

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
