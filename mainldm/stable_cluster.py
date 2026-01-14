import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time
import json
from pathlib import Path
import torch


def load_weights(filepath):
    """Load weights from file (.npy, .npz, or .pth/.pt checkpoint)"""
    import torch
    from pathlib import Path
    
    filepath = Path(filepath)

    if filepath.suffix == '.npy':
        weights = np.load(filepath)

    elif filepath.suffix == '.npz':
        data = np.load(filepath)
        if 'weights' in data:
            weights = data['weights']
        else:
            weights = data[list(data.keys())[0]]

    elif filepath.suffix in ['.pt', '.pth']:
        weights_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        print(f"Loaded data type: {type(weights_data)}")
        
        def tensor_to_numpy(tensor):
            if isinstance(tensor, torch.Tensor):
                if tensor.requires_grad:
                    return tensor.detach().cpu().numpy()
                else:
                    return tensor.cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                return tensor
            else:
                print(f"  Unknown type encountered: {type(tensor)}")
                print(f"  Has detach: {hasattr(tensor, 'detach')}")
                print(f"  Has numpy: {hasattr(tensor, 'numpy')}")
                
                if hasattr(tensor, 'detach'):
                    return tensor.detach().cpu().numpy()
                elif hasattr(tensor, 'numpy'):
                    return tensor.numpy()
                elif hasattr(tensor, 'cpu'):
                    temp = tensor.cpu()
                    if hasattr(temp, 'numpy'):
                        return temp.numpy()
                    elif hasattr(temp, 'detach'):
                        return temp.detach().numpy()
                else:
                    raise ValueError(f"Cannot convert {type(tensor)} to numpy")

        if isinstance(weights_data, dict):
            print(f"Dictionary keys: {list(weights_data.keys())}")
            
            if 'model_state_dict' in weights_data:
                weights_data = weights_data['model_state_dict']
                print("Using 'model_state_dict'")
            elif 'state_dict' in weights_data:
                weights_data = weights_data['state_dict']
                print("Using 'state_dict'")

            if isinstance(weights_data, dict):
                tensors = []
                for name, v in weights_data.items():
                    try:
                        tensor_np = tensor_to_numpy(v)
                        tensors.append(tensor_np.flatten())
                    except Exception as e:
                        print(f"  Warning: Could not convert '{name}': {e}")
                        continue
                
                if tensors:
                    weights = np.concatenate(tensors)
                    print(f"Loaded {len(tensors)} tensors with total {len(weights):,} weights")
                else:
                    raise ValueError("No valid tensors found in checkpoint dictionary")
            else:
                weights = tensor_to_numpy(weights_data)

        else:
            print(f"Single item type: {type(weights_data)}")
            if isinstance(weights_data, (list, tuple)):
                print(f"Found {len(weights_data)} items in list/tuple")
                tensors = []
                for i, item in enumerate(weights_data):
                    try:
                        tensor_np = tensor_to_numpy(item)
                        tensors.append(tensor_np.flatten())
                        print(f"  Item {i}: shape {tensor_np.shape}, size {tensor_np.size}")
                    except Exception as e:
                        print(f"  Warning: Could not convert item {i}: {e}")
                        continue
                
                if tensors:
                    weights = np.concatenate(tensors)
                    print(f"Concatenated {len(tensors)} items with total {len(weights):,} weights")
                else:
                    raise ValueError("No valid tensors found in list/tuple")
            else:
                weights = tensor_to_numpy(weights_data)

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    return weights.flatten()


def calculate_cluster_density(cluster_weights, method='count_over_std'):

    if len(cluster_weights) == 0:
        return 0.0
    
    if method == 'count_over_std':
        std = np.std(cluster_weights)
        if std < 1e-10: 
            return len(cluster_weights) * 1e10
        return len(cluster_weights) / std
    
    elif method == 'negative_std':
        return -np.std(cluster_weights)
    
    elif method == 'count':
        return len(cluster_weights)
    
    elif method == 'inverse_range':
        w_range = np.max(cluster_weights) - np.min(cluster_weights)
        if w_range < 1e-10:
            return 1e10
        return 1.0 / w_range
    
    else:
        raise ValueError(f"Unknown density method: {method}")


def histogram_clustering(weights, n_bins=200, peak_height_ratio=0.05, 
                         min_peak_distance=5, peak_prominence_ratio=0.02,
                         cluster_radius=None, density_method='count_over_std'):

    print("=== Starting Histogram-Based Clustering ===")
    print(f"Density method: {density_method}\n")
    start_time = time.time()
    
    # Create histogram
    hist, bin_edges = np.histogram(weights, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    print(f"Histogram created with {n_bins} bins")
    print(f"Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
 
    peak_height_threshold = np.max(hist) * peak_height_ratio
    peak_prominence_threshold = np.max(hist) * peak_prominence_ratio
    
    peaks, properties = find_peaks(
        hist, 
        height=peak_height_threshold,
        distance=min_peak_distance,
        prominence=peak_prominence_threshold
    )
    
    peak_centers = bin_centers[peaks]
    n_peaks = len(peaks)
    
    print(f"\nDetected {n_peaks} peaks")
    print(f"Peak locations: {peak_centers}")
    if cluster_radius is not None:
        print(f"\nUsing cluster radius: ±{cluster_radius}")
        
        temp_clusters = np.full(len(weights), n_peaks, dtype=int)
        
        for i, w in enumerate(weights):
            distances = np.abs(peak_centers - w)
            min_dist_idx = np.argmin(distances)
            
            if distances[min_dist_idx] <= cluster_radius:
                temp_clusters[i] = min_dist_idx
        
        n_outliers = np.sum(temp_clusters == n_peaks)
        n_temp_clusters = n_peaks + 1 
        
        print(f"Outliers: {n_outliers:,} ({n_outliers/len(weights)*100:.2f}%)")
    else:
        temp_clusters = np.zeros(len(weights), dtype=int)
        for i, w in enumerate(weights):
            temp_clusters[i] = np.argmin(np.abs(peak_centers - w))
        
        n_temp_clusters = n_peaks
    print("\n=== Calculating Cluster Densities ===")
    
    cluster_densities = []
    for label in range(n_temp_clusters):
        mask = temp_clusters == label
        cluster_weights = weights[mask]
        
        is_outlier = (cluster_radius is not None and label == n_peaks)
        
        if is_outlier:
            density = -1e10
        else:
            density = calculate_cluster_density(cluster_weights, method=density_method)
        
        cluster_densities.append((label, density, np.sum(mask), is_outlier))
        
        density_str = "OUTLIERS" if is_outlier else f"{density:.2f}"
        print(f"Cluster {label}: density={density_str}, size={np.sum(mask):,}")
    
    print("\n=== Sorting Clusters by Density (Highest to Lowest) ===")
    
    regular_clusters = [(label, dens, size, False) for label, dens, size, is_out in cluster_densities if not is_out]
    outlier_clusters = [(label, dens, size, True) for label, dens, size, is_out in cluster_densities if is_out]
    
    regular_clusters.sort(key=lambda x: x[1], reverse=True)
    
    sorted_clusters = regular_clusters + outlier_clusters
    
    old_to_new = {}
    
    for new_label, (old_label, density, size, is_outlier) in enumerate(sorted_clusters):
        old_to_new[old_label] = new_label
        
        if is_outlier:
            print(f"Old cluster {old_label} -> New cluster {new_label} (OUTLIERS, size: {size:,})")
        else:
            print(f"Old cluster {old_label} -> New cluster {new_label} (density: {density:.2f}, size: {size:,})")
    
    clusters = np.array([old_to_new[label] for label in temp_clusters])
    
    ordered_peak_centers = []
    ordered_densities = []
    
    for new_label, (old_label, density, size, is_outlier) in enumerate(sorted_clusters):
        if is_outlier:
            ordered_peak_centers.append(0.0) 
            ordered_densities.append(-1e10) 
        else:
            ordered_peak_centers.append(float(peak_centers[old_label]))
            ordered_densities.append(float(density))
    
    n_clusters = n_temp_clusters
    
    elapsed_time = time.time() - start_time
    print(f"\nClustering completed in {elapsed_time:.4f} seconds")
    
    cluster_stats = []
    for new_label in range(n_clusters):
        mask = clusters == new_label
        cluster_weights = weights[mask]
        
        old_label, density, size, is_outlier = sorted_clusters[new_label]
        
        stats = {
            'label': int(new_label),
            'name': 'outliers' if is_outlier else f'cluster_{new_label}',
            'size': int(np.sum(mask)),
            'percentage': float(np.sum(mask) / len(weights) * 100),
            'density': float(density),
            'mean': float(np.mean(cluster_weights)),
            'std': float(np.std(cluster_weights)),
            'min': float(np.min(cluster_weights)),
            'max': float(np.max(cluster_weights)),
            'median': float(np.median(cluster_weights)),
            'is_outlier': is_outlier
        }
        
        if is_outlier:
            stats['peak_location'] = None
            stats['peak_height'] = None
            print(f"\nCluster {new_label} (Outliers):")
        else:
            stats['peak_location'] = float(ordered_peak_centers[new_label])
            peak_idx = np.argmin(np.abs(bin_centers - stats['peak_location']))
            stats['peak_height'] = int(hist[peak_idx])
            print(f"\nCluster {new_label} (Peak at {stats['peak_location']:.6f}):")
        
        print(f"  Density: {stats['density']:.2f}")
        print(f"  Size: {stats['size']:,} ({stats['percentage']:.2f}%)")
        print(f"  Mean: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        
        cluster_stats.append(stats)
    
    results = {
        'weights': weights,
        'clusters': clusters,
        'n_clusters': n_clusters,
        'n_peaks': n_peaks,
        'cluster_centers': ordered_peak_centers, 
        'densities': ordered_densities, 
        'cluster_stats': cluster_stats,
        'histogram': hist,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'peak_indices': peaks,
        'peak_properties': properties,
        'parameters': {
            'n_bins': n_bins,
            'peak_height_ratio': peak_height_ratio,
            'min_peak_distance': min_peak_distance,
            'peak_prominence_ratio': peak_prominence_ratio,
            'cluster_radius': cluster_radius,
            'density_method': density_method
        },
        'timing': {
            'elapsed_seconds': elapsed_time
        }
    }
    
    return results


def save_results_for_main_script(results, output_dir='clustering_output', layer_name='layer'):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    cluster_data = {
        'clusters': results['cluster_centers'],
        'densities': results['densities'],
        'assignments': torch.from_numpy(results['clusters']).long()
    }
    output_path = output_dir / f'{layer_name}_clusters.pth'
    torch.save(cluster_data, output_path)
    print(f"\nSaved clustering data for main script: {output_path}")
    print(f"  Format: {{'clusters': list[{len(cluster_data['clusters'])}], "
          f"'densities': list[{len(cluster_data['densities'])}], "
          f"'assignments': tensor[{cluster_data['assignments'].shape[0]}]}}")
    
    return output_path


def save_results(results, output_dir='clustering_output'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    np.savez(
        output_dir / 'clustering_data.npz',
        weights=results['weights'],
        clusters=results['clusters'],
        cluster_centers=np.array(results['cluster_centers']),
        densities=np.array(results['densities']),
        histogram=results['histogram'],
        bin_edges=results['bin_edges'],
        bin_centers=results['bin_centers'],
        peak_indices=results['peak_indices']
    )
    print(f"\nSaved clustering data to {output_dir / 'clustering_data.npz'}")
    
    metadata = {
        'n_clusters': results['n_clusters'],
        'n_peaks': results['n_peaks'],
        'n_weights': len(results['weights']),
        'cluster_stats': results['cluster_stats'],
        'parameters': results['parameters'],
        'timing': results['timing']
    }
    
    with open(output_dir / 'clustering_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {output_dir / 'clustering_metadata.json'}")
    
    np.savetxt(
        output_dir / 'cluster_assignments.txt',
        results['clusters'],
        fmt='%d',
        header=f'Cluster assignment for each weight (total: {len(results["clusters"])})\n'
               f'Clusters are sorted by DENSITY (highest first)'
    )
    print(f"Saved cluster assignments to {output_dir / 'cluster_assignments.txt'}")
    
    with open(output_dir / 'clustering_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HISTOGRAM-BASED CLUSTERING SUMMARY\n")
        f.write("(Clusters sorted by DENSITY - highest first)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total weights: {len(results['weights']):,}\n")
        f.write(f"Number of clusters: {results['n_clusters']}\n")
        f.write(f"Number of peaks: {results['n_peaks']}\n")
        f.write(f"Clustering time: {results['timing']['elapsed_seconds']:.4f} seconds\n\n")
        
        f.write("PARAMETERS:\n")
        for key, value in results['parameters'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("CLUSTER STATISTICS (sorted by density):\n")
        f.write("-" * 70 + "\n")
        for stats in results['cluster_stats']:
            f.write(f"\nCluster {stats['label']}:\n")
            if stats['is_outlier']:
                f.write(f"  Type: Outliers (outside cluster radius)\n")
            else:
                f.write(f"  Peak location: {stats['peak_location']:.6f}\n")
            f.write(f"  Density: {stats['density']:.4f}\n")
            f.write(f"  Size: {stats['size']:,} ({stats['percentage']:.2f}%)\n")
            f.write(f"  Mean: {stats['mean']:.6f}\n")
            f.write(f"  Std: {stats['std']:.6f}\n")
            f.write(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
            f.write(f"  Median: {stats['median']:.6f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("QUANTIZATION BIT ALLOCATION (4-tier system):\n")
        f.write("=" * 70 + "\n")
        f.write("Based on density sorting, clusters will receive:\n")
        f.write("  - Top 25% (highest density):  4-bit quantization\n")
        f.write("  - Next 25%:                   4-bit quantization\n")
        f.write("  - Next 25%:                   2-bit quantization\n")
        f.write("  - Last 25% (lowest density):  1-bit quantization\n\n")
        
        n_clusters = results['n_clusters']
        quarter = n_clusters // 4
        
        for i, stats in enumerate(results['cluster_stats']):
            if i < quarter:
                bits = 4
            elif i < 2 * quarter:
                bits = 3
            elif i < 3 * quarter:
                bits = 2
            else:
                bits = 1
            
            f.write(f"Cluster {stats['label']}: {bits}-bit "
                   f"(density={stats['density']:.2f}, size={stats['size']:,})\n")
    
    print(f"Saved summary to {output_dir / 'clustering_summary.txt'}")


def visualize_results(results, output_dir='clustering_output'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    weights = results['weights']
    clusters = results['clusters']
    hist = results['histogram']
    bin_centers = results['bin_centers']
    n_clusters = results['n_clusters']
    n_peaks = results['n_peaks']
    n_bins = results['parameters']['n_bins']
    peak_height_threshold = np.max(hist) * results['parameters']['peak_height_ratio']
    cluster_stats = results['cluster_stats']
    densities = results['densities']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(weights, bins=n_bins, alpha=0.6, color='lightblue', edgecolor='black')
    axes[0, 0].plot(bin_centers, hist, 'b-', linewidth=2, label='Histogram')
    
    quarter = n_clusters // 4
    for stats in cluster_stats:
        if not stats['is_outlier']:
            peak_loc = stats['peak_location']
            peak_idx = np.argmin(np.abs(bin_centers - peak_loc))
            
            label = stats['label']
            if label < quarter:
                bits = 4
                color = 'darkgreen'
            elif label < 2 * quarter:
                bits = 3
                color = 'blue'
            elif label < 3 * quarter:
                bits = 2
                color = 'orange'
            else:
                bits = 1
                color = 'red'
            
            axes[0, 0].plot(peak_loc, hist[peak_idx], 'o', markersize=12, 
                           color=color, zorder=5)
            axes[0, 0].text(peak_loc, hist[peak_idx] + np.max(hist)*0.02, 
                           f"C{label}\n{bits}b", ha='center', fontsize=7, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[0, 0].axhline(y=peak_height_threshold, color='r', linestyle='--', 
                       alpha=0.5, label='Peak Threshold')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Peak Detection & Bit Allocation\n(Sorted by density)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    colors_bit = {4: 'darkgreen', 3: 'blue', 2: 'orange', 1: 'red'}
    
    for label in range(n_clusters):
        cluster_weights = weights[clusters == label]
        stats = cluster_stats[label]
        
        if label < quarter:
            bits = 4
        elif label < 2 * quarter:
            bits = 3
        elif label < 3 * quarter:
            bits = 2
        else:
            bits = 1
        
        color = colors_bit.get(bits, 'gray') if not stats['is_outlier'] else 'red'
        
        label_str = f'C{label} - {bits}b ({len(cluster_weights):,})' if not stats['is_outlier'] else f'C{label} - Outliers'
        
        axes[0, 1].hist(cluster_weights, bins=50, alpha=0.6, 
                       label=label_str, color=color, edgecolor='black')
    
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Weight Distribution by Cluster\n(Sorted by density)')
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    x = np.arange(len(weights))
    sample_step = max(1, len(weights) // 10000)
    
    cluster_colors = np.zeros(len(clusters))
    for i in range(len(clusters)):
        label = clusters[i]
        if label < quarter:
            cluster_colors[i] = 4
        elif label < 2 * quarter:
            cluster_colors[i] = 3
        elif label < 3 * quarter:
            cluster_colors[i] = 2
        else:
            cluster_colors[i] = 1
    
    scatter = axes[1, 0].scatter(x[::sample_step], weights[::sample_step], 
                                c=cluster_colors[::sample_step], 
                                cmap='RdYlGn', alpha=0.6, s=1, vmin=1, vmax=4)
    axes[1, 0].set_xlabel('Sample Index (subsampled)')
    axes[1, 0].set_ylabel('Weight Value')
    axes[1, 0].set_title('Clustering Results - Bit Allocation View')
    cbar = plt.colorbar(scatter, ax=axes[1, 0], label='Bits')
    cbar.set_ticks([1, 2, 3, 4])
    axes[1, 0].grid(True, alpha=0.3)
    
    cluster_densities_plot = [stats['density'] for stats in cluster_stats]
    cluster_labels = [stats['label'] for stats in cluster_stats]
    
    bar_colors = []
    for label in cluster_labels:
        if cluster_stats[label]['is_outlier']:
            bar_colors.append('red')
        elif label < quarter:
            bar_colors.append('darkgreen')
        elif label < 2 * quarter:
            bar_colors.append('blue')
        elif label < 3 * quarter:
            bar_colors.append('orange')
        else:
            bar_colors.append('red')
    
    axes[1, 1].bar(cluster_labels, cluster_densities_plot, 
                   color=bar_colors, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Cluster Label (sorted by density)')
    axes[1, 1].set_ylabel('Density Value')
    axes[1, 1].set_title('Cluster Density Distribution\n(Green=4b, Blue=3b, Orange=2b, Red=1b)')
    axes[1, 1].set_xticks(cluster_labels)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for i, (label, density) in enumerate(zip(cluster_labels, cluster_densities_plot)):
        if not cluster_stats[label]['is_outlier']:
            if label < quarter:
                bits = 4
            elif label < 2 * quarter:
                bits = 3
            elif label < 3 * quarter:
                bits = 2
            else:
                bits = 1
            
            axes[1, 1].text(label, density, f'{bits}b', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to {output_dir / 'clustering_visualization.png'}")
    plt.close()


def process_layer_weights(layer_weights_dict, output_dir='clustering_output', **clustering_params):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    print("\n" + "="*80)
    print("PROCESSING MULTIPLE LAYERS")
    print("="*80)
    
    for layer_name, layer_weights in layer_weights_dict.items():
        print(f"\n{'='*80}")
        print(f"Processing layer: {layer_name}")
        print(f"{'='*80}")
        
        if isinstance(layer_weights, torch.Tensor):
            layer_weights = layer_weights.detach().cpu().numpy()
        layer_weights = layer_weights.flatten()
        
        results = histogram_clustering(layer_weights, **clustering_params)
        
        save_results_for_main_script(results, output_dir, layer_name)
        
        all_results[layer_name] = results
    
    print("\n" + "="*80)
    print(f"COMPLETED PROCESSING {len(layer_weights_dict)} LAYERS")
    print("="*80)
    print(f"All cluster files saved to: {output_dir}/")
    print(f"Format: {{layer_name}}_clusters.pth")
    
    return all_results


def load_model_and_extract_weights(model_path, target_layers=None):

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    layer_weights = {}
    
    for name, param in state_dict.items():
        if 'weight' not in name:
            continue
        
        if target_layers is not None:
            if not any(pattern in name for pattern in target_layers):
                continue
        
        clean_name = name.replace('.', '_').replace('/', '_')
        
        if isinstance(param, torch.Tensor):
            layer_weights[clean_name] = param.detach().cpu().numpy()
        else:
            layer_weights[clean_name] = param
        
        print(f"  Extracted: {name} -> {clean_name} (shape: {param.shape})")
    
    print(f"\nExtracted {len(layer_weights)} layers")
    return layer_weights


def cluster_main():
    
    MODE = 1
    
    if MODE == 1:
        print("\n" + "="*80)
        print("MODE 1: Processing single weight file")
        print("="*80 + "\n")
        
        weights = load_weights('./error_dec/stable/weight_params.pth')
        
        params = {
            'n_bins': 500,
            'peak_height_ratio': 0.05,
            'min_peak_distance': 5,
            'peak_prominence_ratio': 0.02,
            'cluster_radius': 0.05,
            'density_method': 'count_over_std' 
        }
        
        output_dir = 'clustering_output'
        
        results = histogram_clustering(weights, **params)
        
        layer_name = 'model_weights'
        save_results_for_main_script(results, output_dir, layer_name)
        
        save_results(results, output_dir)
        
        visualize_results(results, output_dir)
    
    elif MODE == 2:
        print("\n" + "="*80)
        print("MODE 2: Processing multiple layers from model checkpoint")
        print("="*80 + "\n")

        model_path = './mainldm/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'

        target_layers = None
        layer_weights_dict = load_model_and_extract_weights(model_path, target_layers)
        
        params = {
            'n_bins': 500,
            'peak_height_ratio': 0.05,
            'min_peak_distance': 5,
            'peak_prominence_ratio': 0.02,
            'cluster_radius': 0.05,
            'density_method': 'count_over_std'
        }
        
        output_dir = 'clustering_output'
        
        all_results = process_layer_weights(layer_weights_dict, output_dir, **params)
        
        if all_results:
            first_layer = list(all_results.keys())[0]
            print(f"\nCreating visualization for example layer: {first_layer}")
            visualize_results(all_results[first_layer], output_dir)
    
    else:
        raise ValueError(f"Invalid MODE: {MODE}. Choose 1 or 2.")

    
    print("\n" + "=" * 80)
    print("CLUSTERING COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to '{output_dir}/' directory")
    print("\nFiles for main quantization script:")
    print(f"  - {{layer_name}}_clusters.pth (format: {{'clusters': list, 'densities': list, 'assignments': tensor}})")
    print("\nAnalysis files:")
    print(f"  - clustering_data.npz (numpy arrays)")
    print(f"  - clustering_metadata.json (statistics)")
    print(f"  - cluster_assignments.txt (per-weight assignments)")
    print(f"  - clustering_summary.txt (human-readable with bit allocation)")
    print(f"  - clustering_visualization.png (plots)")
    print(f"\nNote: Clusters are sorted by DENSITY (highest first)")
    print("="*80)
    print("\nBit allocation (4-tier system based on density):")
    print("  - Top 25% clusters (highest density):  4-bit quantization")
    print("  - Next 25%:                            3-bit quantization")
    print("  - Next 25%:                            2-bit quantization")
    print("  - Last 25% (lowest density):           1-bit quantization")
    print("="*80)

if __name__ == "__main__":
    cluster_main()