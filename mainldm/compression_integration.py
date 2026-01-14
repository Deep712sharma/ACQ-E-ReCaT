"""
Integration module: Adds actual compression to existing quantization pipeline
Drop-in replacement for integrate_entropy_model.py
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import os

# Import your existing modules
from entropy_model import ClusterEntropyModel, EntropyModelTrainer
from huffman_codec import (
    HuffmanEncoder, 
    LayerCompressor, 
    ModelCompressor,
    create_decompression_wrapper
)

logger = logging.getLogger(__name__)


def apply_entropy_aware_quantization_with_compression(
    q_unet,
    cluster_data: Dict,
    args,
    train_entropy: bool = True,
    num_entropy_iters: int = 500,
    use_compression: bool = True,
    save_compressed: bool = True,
    compressed_model_path: str = "./compressed_model.pkl"
) -> Tuple[Dict, Dict, Optional[Dict]]:
    """
    Enhanced version that adds actual Huffman compression
    
    Args:
        q_unet: Quantized model
        cluster_data: Cluster information
        args: Arguments
        train_entropy: Whether to train entropy models
        num_entropy_iters: Entropy training iterations
        use_compression: Whether to actually compress (vs just estimate)
        save_compressed: Whether to save compressed model to disk
        compressed_model_path: Where to save compressed model
        
    Returns:
        cluster_quantizers: Quantizer objects
        bitrate_info: Bitrate information
        compressed_layers: Compressed layer data (if use_compression=True)
    """
    logger.info("\n" + "="*80)
    logger.info("APPLYING ENTROPY-AWARE CLUSTER QUANTIZATION")
    if use_compression:
        logger.info("WITH ACTUAL HUFFMAN COMPRESSION")
    else:
        logger.info("(Estimation only - no actual compression)")
    logger.info("="*80)
    
    from integrate_entropy_model import (
        ClusterBasedQuantizer,
        find_matching_layer
    )
    
    layer_clusters = cluster_data['layer_clusters']
    cluster_quantizers = {}
    bitrate_info = {}
    
    # PHASE 1: Create quantizers and train entropy models
    logger.info("\n=== PHASE 1: Quantization and Entropy Training ===")
    
    for name, module in q_unet.named_modules():
        if not hasattr(module, 'org_weight'):
            continue
        
        cluster_key = find_matching_layer(name, layer_clusters)
        if cluster_key is None:
            continue
        
        logger.info(f"\nProcessing layer: {name}")
        
        org_weight = module.org_weight.cuda()
        cluster_info = layer_clusters[cluster_key]
        
        # Create quantizer
        quantizer = ClusterBasedQuantizer(
            clusters=cluster_info['clusters'],
            densities=cluster_info['densities'],
            assignments=cluster_info['assignments'],
            num_clusters=len(cluster_info['clusters'])
        )
        
        # Train entropy model
        if train_entropy:
            logger.info(f"  Training entropy model...")
            metrics = quantizer.train_entropy_model(
                org_weight,
                num_iterations=num_entropy_iters
            )
            logger.info(f"  ✓ Trained: {metrics['bits_per_weight']:.3f} bits/weight (estimated)")
        
        # Quantize weights
        quantized_weight = quantizer.quantize_weights(org_weight)
        
        # Calculate theoretical bitrate
        bitrate = quantizer.calculate_bitrate(quantized_weight)
        
        cluster_quantizers[name] = quantizer
        
        num_weights = org_weight.numel()
        bitrate_info[name] = {
            'bitrate': bitrate,
            'num_weights': num_weights,
            'total_bits': bitrate * num_weights
        }
        
        logger.info(f"  ✓ Quantized: {num_weights:,} weights")
        logger.info(f"  Theoretical bitrate: {bitrate:.3f} bits/weight")
    
    # PHASE 2: Actual compression (if enabled)
    compressed_layers = None
    
    if use_compression:
        logger.info("\n=== PHASE 2: Huffman Compression ===")
        
        model_compressor = ModelCompressor(cluster_quantizers)
        compressed_layers = model_compressor.compress_model(q_unet)
        
        # Update bitrate info with actual compression stats
        for name, compressed_data in compressed_layers.items():
            metadata = compressed_data['metadata']
            actual_bits = len(compressed_data['compressed_bytes']) * 8
            actual_bitrate = actual_bits / metadata['num_symbols']
            
            bitrate_info[name]['actual_bitrate'] = actual_bitrate
            bitrate_info[name]['actual_total_bits'] = actual_bits
            bitrate_info[name]['compressed_bytes'] = len(compressed_data['compressed_bytes'])
            
            logger.info(f"\n{name}:")
            logger.info(f"  Theoretical: {bitrate_info[name]['bitrate']:.3f} bits/weight")
            logger.info(f"  Actual: {actual_bitrate:.3f} bits/weight")
            logger.info(f"  Difference: {abs(bitrate_info[name]['bitrate'] - actual_bitrate):.3f} bits")
        
        # Save compressed model
        if save_compressed:
            model_compressor.save_compressed_model(compressed_model_path)
            logger.info(f"\n✓ Compressed model saved to: {compressed_model_path}")
    
    # PHASE 3: Summary statistics
    logger.info("\n=== PHASE 3: Summary Statistics ===")
    
    total_weights = sum(info['num_weights'] for info in bitrate_info.values())
    total_theoretical_bits = sum(info['total_bits'] for info in bitrate_info.values())
    
    if total_weights > 0:
        avg_theoretical_bitrate = total_theoretical_bits / total_weights
        theoretical_compression = 32.0 / avg_theoretical_bitrate
        
        logger.info(f"\nTheoretical (Shannon Entropy):")
        logger.info(f"  Average bitrate: {avg_theoretical_bitrate:.3f} bits/weight")
        logger.info(f"  Compression ratio: {theoretical_compression:.2f}x (vs FP32)")
        logger.info(f"  Model size: {total_theoretical_bits/8/1024/1024:.2f} MB")
    
    if use_compression and compressed_layers:
        total_actual_bits = sum(info.get('actual_total_bits', 0) for info in bitrate_info.values())
        avg_actual_bitrate = total_actual_bits / total_weights if total_weights > 0 else 0
        actual_compression = 32.0 / avg_actual_bitrate if avg_actual_bitrate > 0 else 0
        
        logger.info(f"\nActual (Huffman Coding):")
        logger.info(f"  Average bitrate: {avg_actual_bitrate:.3f} bits/weight")
        logger.info(f"  Compression ratio: {actual_compression:.2f}x (vs FP32)")
        logger.info(f"  Model size: {total_actual_bits/8/1024/1024:.2f} MB")
        logger.info(f"  Huffman overhead: {((avg_actual_bitrate - avg_theoretical_bitrate) / avg_theoretical_bitrate * 100):.2f}%")
    
    logger.info("="*80 + "\n")
    
    return cluster_quantizers, bitrate_info, compressed_layers


def benchmark_compression_decompression(
    compressed_layers: Dict,
    cluster_quantizers: Dict,
    num_trials: int = 10
):
    """
    Benchmark compression/decompression speed
    
    Args:
        compressed_layers: Compressed layer data
        cluster_quantizers: Quantizer objects
        num_trials: Number of trials for timing
    """
    import time
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARKING COMPRESSION/DECOMPRESSION SPEED")
    logger.info("="*80)
    
    model_compressor = ModelCompressor(cluster_quantizers)
    model_compressor.compressed_layers = compressed_layers
    
    for layer_name in list(compressed_layers.keys())[:3]:  # Test first 3 layers
        logger.info(f"\nLayer: {layer_name}")
        
        # Benchmark decompression
        decompress_times = []
        for _ in range(num_trials):
            start = time.time()
            weights = model_compressor.decompress_layer(layer_name)
            torch.cuda.synchronize()
            decompress_times.append(time.time() - start)
        
        avg_decompress = np.mean(decompress_times)
        std_decompress = np.std(decompress_times)
        
        metadata = compressed_layers[layer_name]['metadata']
        num_weights = metadata['num_symbols']
        
        logger.info(f"  Decompression time: {avg_decompress*1000:.2f} ± {std_decompress*1000:.2f} ms")
        logger.info(f"  Throughput: {num_weights/avg_decompress/1e6:.2f} M weights/sec")
        logger.info(f"  Weights: {num_weights:,}")
    
    logger.info("="*80 + "\n")


def compare_compression_methods(
    q_unet,
    cluster_quantizers: Dict,
    sample_layers: int = 5
):
    """
    Compare theoretical Shannon entropy vs actual Huffman compression
    
    Args:
        q_unet: Quantized model
        cluster_quantizers: Quantizer objects
        sample_layers: Number of layers to compare
    """
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: SHANNON ENTROPY vs HUFFMAN CODING")
    logger.info("="*80)
    
    results = []
    layer_count = 0
    
    for name, module in q_unet.named_modules():
        if layer_count >= sample_layers:
            break
        
        if not hasattr(module, 'org_weight'):
            continue
        
        if name not in cluster_quantizers:
            continue
        
        logger.info(f"\nLayer: {name}")
        
        quantizer = cluster_quantizers[name]
        org_weight = module.org_weight.cuda()
        
        # Get quantized weights
        quantized_weight = quantizer.quantize_weights(org_weight)
        
        # Theoretical (Shannon entropy)
        theoretical_bitrate = quantizer.calculate_bitrate(quantized_weight)
        
        # Actual (Huffman)
        compressor = LayerCompressor(quantizer, cluster_id=0)
        compressed_data = compressor.compress(org_weight)
        actual_bits = len(compressed_data['compressed_bytes']) * 8
        actual_bitrate = actual_bits / org_weight.numel()
        
        # Calculate difference
        huffman_overhead = actual_bitrate - theoretical_bitrate
        overhead_percent = (huffman_overhead / theoretical_bitrate) * 100
        
        results.append({
            'layer': name,
            'theoretical': theoretical_bitrate,
            'actual': actual_bitrate,
            'overhead': huffman_overhead,
            'overhead_pct': overhead_percent
        })
        
        logger.info(f"  Shannon entropy: {theoretical_bitrate:.3f} bits/weight")
        logger.info(f"  Huffman actual: {actual_bitrate:.3f} bits/weight")
        logger.info(f"  Overhead: {huffman_overhead:.3f} bits ({overhead_percent:.2f}%)")
        
        layer_count += 1
    
    # Summary
    avg_theoretical = np.mean([r['theoretical'] for r in results])
    avg_actual = np.mean([r['actual'] for r in results])
    avg_overhead_pct = np.mean([r['overhead_pct'] for r in results])
    
    logger.info("\n" + "-"*80)
    logger.info("SUMMARY:")
    logger.info(f"  Average Shannon entropy: {avg_theoretical:.3f} bits/weight")
    logger.info(f"  Average Huffman actual: {avg_actual:.3f} bits/weight")
    logger.info(f"  Average overhead: {avg_overhead_pct:.2f}%")
    logger.info(f"  Note: Huffman uses whole bits, causing small overhead vs theoretical entropy")
    logger.info("="*80 + "\n")
    
    return results


def load_and_use_compressed_model(
    compressed_model_path: str,
    q_unet,
    test_inference: bool = True
):
    """
    Load compressed model and optionally test inference
    
    Args:
        compressed_model_path: Path to compressed model
        q_unet: Model to load weights into
        test_inference: Whether to test inference
        
    Returns:
        model_compressor: ModelCompressor instance with loaded data
    """
    logger.info("\n" + "="*80)
    logger.info("LOADING COMPRESSED MODEL")
    logger.info("="*80)
    
    # Load compressed data
    compressed_layers, quantizer_configs = ModelCompressor.load_compressed_model(
        compressed_model_path
    )
    
    logger.info(f"Loaded {len(compressed_layers)} compressed layers")
    
    # Recreate quantizers
    from integrate_entropy_model import ClusterBasedQuantizer
    
    cluster_quantizers = {}
    for name, config in quantizer_configs.items():
        quantizer = ClusterBasedQuantizer(
            clusters=config['clusters'],
            densities=config['densities'],
            assignments=config['assignments'],
            num_clusters=config['num_clusters']
        )
        quantizer.bit_allocation = config['bit_allocation'].cuda()
        cluster_quantizers[name] = quantizer
    
    # Create model compressor
    model_compressor = ModelCompressor(cluster_quantizers)
    model_compressor.compressed_layers = compressed_layers
    
    # Optionally wrap model for decompression during inference
    if test_inference:
        logger.info("\nWrapping model with decompression...")
        create_decompression_wrapper(q_unet, compressed_layers, cluster_quantizers)
        logger.info("✓ Model ready for inference with on-the-fly decompression")
    
    logger.info("="*80 + "\n")
    
    return model_compressor


# Convenience function for main script
def quick_compress_model(q_unet, cluster_data, args):
    """
    One-line function to compress model
    
    Usage in main.py:
        compressed_layers = quick_compress_model(q_unet, cluster_data, args)
    """
    cluster_quantizers, bitrate_info, compressed_layers = \
        apply_entropy_aware_quantization_with_compression(
            q_unet,
            cluster_data,
            args,
            train_entropy=args.train_entropy,
            num_entropy_iters=args.entropy_iters,
            use_compression=True,
            save_compressed=True,
            compressed_model_path=args.entropy_models_path.replace('.pth', '_compressed.pkl')
        )
    
    return compressed_layers, cluster_quantizers, bitrate_info