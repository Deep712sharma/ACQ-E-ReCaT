"""
Fixed average.py that works with both cluster quantization and entropy models
Compatible with integrate_entropy_model.py
"""
import heapq
from collections import Counter
import numpy as np
import torch
import json
import os


def average(q_unet, args, bitrate_info=None):
    """
    CLUSTER-AWARE + ENTROPY-AWARE analysis of the quantized diffusion model.
    
    Args:
        q_unet: Quantized model
        args: Arguments
        bitrate_info: Optional dict with entropy model bitrates per layer
    
    Computes:
      - Average quantized bitwidth (considering cluster-based mixed precision)
      - Entropy-coded bitwidth (if entropy models are used)
      - Weight statistics and per-layer details
    """

    print("=" * 80)
    print("CLUSTER-AWARE QUANTIZED DIFFUSION MODEL ANALYSIS")
    if bitrate_info is not None:
        print("WITH ENTROPY MODEL INTEGRATION")
    print("=" * 80)

    results = {
        'total_params': 0,
        'total_bits': 0,
        'entropy_quantized_params': 0,
        'entropy_quantized_bits': 0,
        'cluster_quantized_params': 0,
        'cluster_quantized_bits': 0,
        'base_quantized_params': 0,
        'base_quantized_bits': 0,
        'fp32_params': 0,
        'weight_stats': {'sum': 0, 'count': 0, 'min': float('inf'), 'max': float('-inf')},
        'layer_details': [],
        'huffman_avg_bitwidth': None
    }

    all_quantized_values = []

    # ----------------------------------------------------
    # Traverse all modules in the quantized UNet
    # ----------------------------------------------------
    for name, module in q_unet.named_modules():
        # Check if module has weights
        if not hasattr(module, 'org_weight'):
            continue
            
        weight = module.org_weight
        num_params = weight.numel()
        
        # Get weight values for statistics
        weight_values = weight.detach().cpu().float().numpy().flatten()
        
        # Determine quantization type (priority order)
        # 1. Entropy-aware quantization (highest priority)
        # 2. Cluster-based quantization
        # 3. Base quantization
        # 4. FP32 (unquantized)
        
        # FIXED: Check for use_entropy_quant (set by integrate_entropy_model.py)
        is_entropy_quant = (hasattr(module, 'use_entropy_quant') and 
                           module.use_entropy_quant and 
                           hasattr(module, 'cluster_quantizer') and
                           module.cluster_quantizer is not None)
        
        is_cluster_quant = (hasattr(module, 'cluster_quantizer') and 
                           module.cluster_quantizer is not None and
                           not is_entropy_quant)  # Only if not already entropy quant
        
        is_base_quant = (hasattr(module, 'weight_quantizer') and 
                        module.weight_quantizer is not None and
                        not is_entropy_quant and 
                        not is_cluster_quant)
        
        if is_entropy_quant:
            # ENTROPY-AWARE QUANTIZATION (uses learned bitrate)
            quantizer = module.cluster_quantizer
            
            # Get bitrate from bitrate_info if available, otherwise calculate
            if bitrate_info is not None and name in bitrate_info:
                bitrate = bitrate_info[name]['bitrate']
            else:
                # Calculate from quantizer
                bitrate = quantizer.get_average_bitwidth()
            
            total_layer_bits = bitrate * num_params
            
            all_quantized_values.extend(weight_values.tolist())
            
            # Update results
            results['entropy_quantized_params'] += num_params
            results['entropy_quantized_bits'] += total_layer_bits
            results['total_params'] += num_params
            results['total_bits'] += total_layer_bits
            
            # Get cluster info
            num_clusters = len(quantizer.clusters)
            is_trained = quantizer.is_trained
            
            layer_info = {
                'name': name,
                'type': 'EntropyAware',
                'num_params': num_params,
                'bitrate': float(bitrate),
                'total_bits': float(total_layer_bits),
                'num_clusters': num_clusters,
                'entropy_trained': is_trained,
                'weight_mean_abs': float(np.mean(np.abs(weight_values))),
                'weight_std': float(np.std(weight_values))
            }
            
            print(f"\n✓ ENTROPY: {name}")
            print(f"    Params: {num_params:,}, Bitrate: {bitrate:.3f} bits/weight")
            print(f"    Clusters: {num_clusters}, Trained: {is_trained}")
            print(f"    Total bits: {total_layer_bits:,.0f}")
            
        elif is_cluster_quant:
            # CLUSTER-BASED QUANTIZATION (without entropy model)
            quantizer = module.cluster_quantizer
            
            all_quantized_values.extend(weight_values.tolist())
            
            # Calculate actual bits used (mixed precision)
            avg_bitwidth = quantizer.get_average_bitwidth()
            total_layer_bits = avg_bitwidth * num_params
            
            # Get bit allocation per cluster
            bit_allocation_dict = {}
            for cluster_id in range(len(quantizer.clusters)):
                cluster_bits = quantizer.bit_allocation[cluster_id].item()
                bit_allocation_dict[cluster_id] = float(cluster_bits)
            
            # Update results
            results['cluster_quantized_params'] += num_params
            results['cluster_quantized_bits'] += total_layer_bits
            results['total_params'] += num_params
            results['total_bits'] += total_layer_bits
            
            layer_info = {
                'name': name,
                'type': 'ClusterQuantized',
                'num_params': num_params,
                'avg_bitwidth': float(avg_bitwidth),
                'total_bits': float(total_layer_bits),
                'num_clusters': len(quantizer.clusters),
                'bit_allocation': bit_allocation_dict,
                'weight_mean_abs': float(np.mean(np.abs(weight_values))),
                'weight_std': float(np.std(weight_values))
            }
            
            print(f"\n✓ CLUSTER: {name}")
            print(f"    Params: {num_params:,}, Avg bits: {avg_bitwidth:.2f}")
            print(f"    Clusters: {len(quantizer.clusters)}, Total bits: {total_layer_bits:,.0f}")
            
        elif is_base_quant:
            # BASE QUANTIZATION
            weight_bitwidth = args.weight_bit
            if hasattr(module.weight_quantizer, 'n_bits'):
                weight_bitwidth = module.weight_quantizer.n_bits
            
            all_quantized_values.extend(weight_values.tolist())
            
            total_layer_bits = num_params * weight_bitwidth
            
            results['base_quantized_params'] += num_params
            results['base_quantized_bits'] += total_layer_bits
            results['total_params'] += num_params
            results['total_bits'] += total_layer_bits
            
            layer_info = {
                'name': name,
                'type': 'BaseQuantized',
                'num_params': num_params,
                'weight_bitwidth': weight_bitwidth,
                'total_bits': float(total_layer_bits),
                'weight_mean_abs': float(np.mean(np.abs(weight_values))),
                'weight_std': float(np.std(weight_values))
            }
            
            print(f"\n→ BASE: {name}")
            print(f"    Params: {num_params:,}, Bits: {weight_bitwidth}")
            
        else:
            # FP32 (unquantized)
            # Count as 32-bit for fair comparison
            total_layer_bits = num_params * 32
            
            results['fp32_params'] += num_params
            results['total_params'] += num_params
            results['total_bits'] += total_layer_bits
            
            layer_info = {
                'name': name,
                'type': 'FP32',
                'num_params': num_params,
                'weight_bitwidth': 32,
                'total_bits': float(total_layer_bits),
                'weight_mean_abs': float(np.mean(np.abs(weight_values))),
                'weight_std': float(np.std(weight_values))
            }
            
            print(f"\n○ FP32: {name}")
            print(f"    Params: {num_params:,}, Bits: 32")
        
        # Update weight stats
        results['weight_stats']['sum'] += np.sum(np.abs(weight_values))
        results['weight_stats']['count'] += len(weight_values)
        results['weight_stats']['min'] = min(results['weight_stats']['min'], weight_values.min())
        results['weight_stats']['max'] = max(results['weight_stats']['max'], weight_values.max())
        
        results['layer_details'].append(layer_info)

    # ----------------------------------------------------
    # Compute Average Bitwidth
    # ----------------------------------------------------
    avg_bitwidth = results['total_bits'] / results['total_params'] if results['total_params'] > 0 else 0
    avg_weight_value = results['weight_stats']['sum'] / results['weight_stats']['count'] if results['weight_stats']['count'] > 0 else 0
    
    # Quantized-only average (excluding FP32 layers)
    quantized_params = (results['entropy_quantized_params'] + 
                       results['cluster_quantized_params'] + 
                       results['base_quantized_params'])
    quantized_bits = (results['entropy_quantized_bits'] + 
                     results['cluster_quantized_bits'] + 
                     results['base_quantized_bits'])
    avg_bitwidth_quantized_only = quantized_bits / quantized_params if quantized_params > 0 else 0

    # ----------------------------------------------------
    # Display Final Results
    # ----------------------------------------------------
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total Parameters:                    {results['total_params']:,}")
    
    if results['entropy_quantized_params'] > 0:
        pct = 100 * results['entropy_quantized_params'] / results['total_params']
        avg_entropy_bits = results['entropy_quantized_bits'] / results['entropy_quantized_params']
        print(f"  - Entropy-aware quantized:         {results['entropy_quantized_params']:,} ({pct:.1f}%)")
        print(f"    Avg bitrate:                     {avg_entropy_bits:.3f} bits/weight")
    
    if results['cluster_quantized_params'] > 0:
        pct = 100 * results['cluster_quantized_params'] / results['total_params']
        avg_cluster_bits = results['cluster_quantized_bits'] / results['cluster_quantized_params']
        print(f"  - Cluster-quantized:               {results['cluster_quantized_params']:,} ({pct:.1f}%)")
        print(f"    Avg bitrate:                     {avg_cluster_bits:.3f} bits/weight")
    
    if results['base_quantized_params'] > 0:
        pct = 100 * results['base_quantized_params'] / results['total_params']
        avg_base_bits = results['base_quantized_bits'] / results['base_quantized_params']
        print(f"  - Base-quantized:                  {results['base_quantized_params']:,} ({pct:.1f}%)")
        print(f"    Avg bitrate:                     {avg_base_bits:.3f} bits/weight")
    
    if results['fp32_params'] > 0:
        pct = 100 * results['fp32_params'] / results['total_params']
        print(f"  - FP32 (unquantized):              {results['fp32_params']:,} ({pct:.1f}%)")
    
    print(f"\nAverage Bitwidth (all layers):       {avg_bitwidth:.2f} bits")
    print(f"Average Bitwidth (quantized only):   {avg_bitwidth_quantized_only:.2f} bits")
    
    if avg_bitwidth_quantized_only > 0:
        compression = 32.0 / avg_bitwidth_quantized_only
        print(f"Compression Ratio (from FP32):       {compression:.2f}×")
    
    print(f"\nMean |Weight|:                       {avg_weight_value:.6f}")
    print(f"Weight Range:                        [{results['weight_stats']['min']:.6f}, {results['weight_stats']['max']:.6f}]")
    print(f"Configured Bits (W/A):               {args.weight_bit}/{args.act_bit}")
    print("=" * 80)

    # Save summary
    output_file = "summary/quantization_analysis.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    summary = {
        'avg_bitwidth_all': float(avg_bitwidth),
        'avg_bitwidth_quantized_only': float(avg_bitwidth_quantized_only),
        'avg_weight_value': float(avg_weight_value),
        'compression_ratio': float(32.0 / avg_bitwidth_quantized_only) if avg_bitwidth_quantized_only > 0 else None,
        'total_params': int(results['total_params']),
        'entropy_quantized_params': int(results['entropy_quantized_params']),
        'cluster_quantized_params': int(results['cluster_quantized_params']),
        'base_quantized_params': int(results['base_quantized_params']),
        'fp32_params': int(results['fp32_params']),
        'entropy_quantized_bits': float(results['entropy_quantized_bits']),
        'cluster_quantized_bits': float(results['cluster_quantized_bits']),
        'base_quantized_bits': float(results['base_quantized_bits']),
        'weight_stats': {
            'mean_abs': float(avg_weight_value),
            'min': float(results['weight_stats']['min']),
            'max': float(results['weight_stats']['max'])
        },
        'layer_count': int(len(results['layer_details'])),
        'layer_details': results['layer_details']
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    return avg_bitwidth_quantized_only, avg_weight_value, results