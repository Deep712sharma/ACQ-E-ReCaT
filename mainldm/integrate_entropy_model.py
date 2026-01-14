"""
Integration module for entropy model with cluster-based quantization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from entropy_model import ClusterEntropyModel, EntropyModelTrainer
logger = logging.getLogger(__name__)


class ClusterBasedQuantizer:
    """
    Cluster-based quantizer with learned entropy model for optimal bit allocation
    """
    def __init__(self, 
                 clusters,
                 densities,
                 assignments,
                 num_clusters: int):
        self.clusters = clusters
        self.densities = densities
        self.assignments = assignments
        self.num_clusters = num_clusters
        
        # Initialize bit allocation based on density
        self.bit_allocation = self._initialize_bit_allocation()
        
        # Entropy model (will be created later)
        self.entropy_model = None
        self.is_trained = False
        
    def _initialize_bit_allocation(self) -> torch.Tensor:
        """
        Initialize bit allocation for each cluster based on density
        Higher density clusters get more bits
        """
        bit_allocation = torch.zeros(self.num_clusters)
        
        # Sort densities to determine quartiles
        sorted_densities = sorted(enumerate(self.densities), 
                                 key=lambda x: x[1], reverse=True)
        
        quarter = self.num_clusters // 4
        
        for i, (cluster_id, density) in enumerate(sorted_densities):
            if i < quarter:
                bit_allocation[cluster_id] = 4  # Top 25%
            elif i < 2 * quarter:
                bit_allocation[cluster_id] = 3  # Next 25%
            elif i < 3 * quarter:
                bit_allocation[cluster_id] = 2  # Next 25%
            else:
                bit_allocation[cluster_id] = 1  # Last 25%
        
        return bit_allocation.cuda()
    
    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Quantize weights using cluster-based mixed precision
        
        Args:
            weights: Original weights
            
        Returns:
            quantized_weights: Quantized weights
        """
        weights_flat = weights.flatten()
        assignments_flat = self.assignments.flatten()
        quantized_flat = torch.zeros_like(weights_flat)
        
        for cluster_id in range(self.num_clusters):
            mask = (assignments_flat == cluster_id)
            cluster_weights = weights_flat[mask]
            
            if cluster_weights.numel() == 0:
                continue
            
            # Get bit allocation for this cluster
            n_bits = int(self.bit_allocation[cluster_id].item())
            n_levels = 2 ** n_bits
            
            # Uniform quantization
            w_min = cluster_weights.min()
            w_max = cluster_weights.max()
            
            if w_max - w_min < 1e-8:
                quantized_flat[mask] = cluster_weights
            else:
                scale = (w_max - w_min) / (n_levels - 1)
                q_weights = torch.round((cluster_weights - w_min) / scale) * scale + w_min
                quantized_flat[mask] = q_weights
        
        return quantized_flat.reshape(weights.shape)
    
    def train_entropy_model(self, 
                           original_weights: torch.Tensor,
                           num_iterations: int = 500) -> Dict:
        """
        Train entropy model for this layer
        
        Args:
            original_weights: Original FP32 weights
            num_iterations: Training iterations
            
        Returns:
            metrics: Training metrics
        """
        
        
        # Create entropy model
        self.entropy_model = ClusterEntropyModel(
            num_clusters=self.num_clusters,
            max_levels=256
        ).cuda()
        
        # Quantize weights
        quantized_weights = self.quantize_weights(original_weights)
        
        # Train entropy model
        trainer = EntropyModelTrainer(self.entropy_model, lr=1e-3)
        metrics = trainer.train(
            quantized_weights,
            self.assignments.cuda(),
            self.bit_allocation,
            num_iterations
        )
        
        self.is_trained = True
        return metrics
    
    def calculate_bitrate(self, quantized_weights: torch.Tensor) -> float:
        """
        Calculate actual bitrate using entropy model
        """
        if not self.is_trained or self.entropy_model is None:
            # Fallback to theoretical bitrate
            return self._calculate_theoretical_bitrate()
        
        from entropy_model import calculate_actual_bitrate
        
        return calculate_actual_bitrate(
            quantized_weights,
            self.assignments.cuda(),
            self.bit_allocation,
            self.entropy_model
        )
    
    def _calculate_theoretical_bitrate(self) -> float:
        """
        Calculate theoretical bitrate from bit allocation
        """
        assignments_flat = self.assignments.flatten()
        total_bits = 0
        total_weights = assignments_flat.numel()
        
        for cluster_id in range(self.num_clusters):
            mask = (assignments_flat == cluster_id)
            count = mask.sum().item()
            bits = self.bit_allocation[cluster_id].item()
            total_bits += bits * count
        
        return total_bits / total_weights if total_weights > 0 else 0
    
    def get_average_bitwidth(self) -> float:
        """Calculate average bit-width"""
        return self._calculate_theoretical_bitrate()


def calculate_average_bitwidth(model, cluster_quantizers: Dict) -> float:
    """
    Calculate average bitwidth across all quantized layers
    
    Args:
        model: Quantized model
        cluster_quantizers: Dictionary of quantizers per layer
        
    Returns:
        avg_bitwidth: Average bitwidth
    """
    total_weights = 0
    total_bits = 0
    
    for name, quantizer in cluster_quantizers.items():
        # Find corresponding module
        module = None
        for n, m in model.named_modules():
            if n == name and hasattr(m, 'org_weight'):
                module = m
                break
        
        if module is None:
            continue
        
        num_weights = module.org_weight.numel()
        avg_bits = quantizer.get_average_bitwidth()
        
        total_weights += num_weights
        total_bits += avg_bits * num_weights
    
    if total_weights > 0:
        return total_bits / total_weights
    return 0


def find_matching_layer(layer_name: str, layer_clusters: Dict) -> Optional[str]:
    """
    Find matching cluster layer name for a model layer
    Handles different naming conventions
    
    Args:
        layer_name: Name from model
        layer_clusters: Dictionary of cluster data
        
    Returns:
        Matching cluster key or None
    """
    # Exact match
    if layer_name in layer_clusters:
        return layer_name
    
    # Try different naming patterns
    patterns = [
        layer_name.replace('model.', ''),  # Remove 'model.' prefix
        layer_name.replace('org_module.', ''),  # Remove 'org_module.' prefix
        'model.' + layer_name,  # Add 'model.' prefix
        layer_name.split('.')[-1],  # Just the last component
    ]
    
    for pattern in patterns:
        if pattern in layer_clusters:
            return pattern
    
    # Partial match - check if layer_name is contained in any cluster key
    for cluster_key in layer_clusters.keys():
        if layer_name in cluster_key or cluster_key in layer_name:
            return cluster_key
    
    return None


def apply_entropy_aware_quantization(q_unet, 
                                     cluster_data: Dict,
                                     args,
                                     train_entropy: bool = True,
                                     num_entropy_iters: int = 500) -> Tuple:
    """
    Apply cluster quantization with learned entropy models
    
    Args:
        q_unet: Quantized model
        cluster_data: Cluster information from clustering
        args: Arguments
        train_entropy: Whether to train entropy models
        num_entropy_iters: Iterations for entropy model training
        
    Returns:
        cluster_quantizers: Dictionary of quantizers per layer
        bitrate_info: Bitrate information per layer
    """
    logger.info("\n" + "="*80)
    logger.info("APPLYING ENTROPY-AWARE CLUSTER QUANTIZATION")
    logger.info("="*80)
    
    layer_clusters = cluster_data['layer_clusters']
    cluster_quantizers = {}
    bitrate_info = {}
    
    total_weights = 0
    total_bits = 0
    
    # Debug: print all module names to find matching pattern
    logger.info("\nSearching for quantizable layers...")
    all_module_names = [name for name, module in q_unet.named_modules() if hasattr(module, 'org_weight')]
    logger.info(f"Found {len(all_module_names)} modules with org_weight:")
    for name in all_module_names[:5]:  # Show first 5
        logger.info(f"  - {name}")
    if len(all_module_names) > 5:
        logger.info(f"  ... and {len(all_module_names) - 5} more")
    
    logger.info(f"\nCluster files loaded for {len(layer_clusters)} layers:")
    for name in list(layer_clusters.keys())[:5]:  # Show first 5
        logger.info(f"  - {name}")
    if len(layer_clusters) > 5:
        logger.info(f"  ... and {len(layer_clusters) - 5} more")
    
    for name, module in q_unet.named_modules():
        if not hasattr(module, 'org_weight'):
            continue
        
        # Try to find matching cluster data
        cluster_key = find_matching_layer(name, layer_clusters)
        
        if cluster_key is None:
            continue
        
        logger.info(f"\nProcessing layer: {name}")
        if cluster_key != name:
            logger.info(f"  Matched with cluster file: {cluster_key}")
        
        org_weight = module.org_weight.cuda()
        cluster_info = layer_clusters[cluster_key]
        
        # Create cluster-based quantizer
        quantizer = ClusterBasedQuantizer(
            clusters=cluster_info['clusters'],
            densities=cluster_info['densities'],
            assignments=cluster_info['assignments'],
            num_clusters=len(cluster_info['clusters'])
        )
        
        # Train entropy model if requested
        if train_entropy:
            logger.info(f"  Training entropy model...")
            metrics = quantizer.train_entropy_model(
                org_weight,
                num_iterations=num_entropy_iters
            )
            logger.info(f"  ✓ Entropy model trained: {metrics['bits_per_weight']:.3f} bits/weight")
        
        # Apply quantization
        quantized_weight = quantizer.quantize_weights(org_weight)
        
        # Calculate actual bitrate
        bitrate = quantizer.calculate_bitrate(quantized_weight)
        
        # Store quantizer
        cluster_quantizers[name] = quantizer
        
        # Update module to use entropy-aware quantization
        module.cluster_quantizer = quantizer
        module.use_entropy_quant = True
        
        # Store bitrate information
        num_weights = org_weight.numel()
        bitrate_info[name] = {
            'bitrate': bitrate,
            'num_weights': num_weights,
            'total_bits': bitrate * num_weights
        }
        
        total_weights += num_weights
        total_bits += bitrate * num_weights
        
        logger.info(f"  ✓ Layer quantized")
        logger.info(f"    Bitrate: {bitrate:.3f} bits/weight")
        logger.info(f"    Weights: {num_weights:,}")
        logger.info(f"    Total bits: {bitrate * num_weights:,.0f}")
    
    # Calculate overall statistics
    if total_weights > 0:
        avg_bitrate = total_bits / total_weights
        compression_ratio = 32.0 / avg_bitrate
    else:
        avg_bitrate = 0
        compression_ratio = 0
        logger.warning("\n⚠️  WARNING: No layers were quantized!")
        logger.warning("This likely means layer names in cluster files don't match model layer names.")
        logger.warning("Please check the debug output above to verify layer name matching.")
    
    logger.info(f"\n" + "="*80)
    logger.info(f"ENTROPY-AWARE QUANTIZATION SUMMARY:")
    logger.info(f"  Layers quantized: {len(cluster_quantizers)}")
    logger.info(f"  Total weights: {total_weights:,}")
    logger.info(f"  Total bits: {total_bits:,.0f}")
    logger.info(f"  Average bitrate: {avg_bitrate:.3f} bits/weight")
    if avg_bitrate > 0:
        logger.info(f"  Compression ratio: {compression_ratio:.2f}x (from FP32)")
    logger.info("="*80 + "\n")
    
    return cluster_quantizers, bitrate_info


def modify_forward_for_entropy_quant(module, quantizer):
    """
    Modify module's forward pass to use entropy-aware quantization
    
    Args:
        module: Layer module
        quantizer: ClusterBasedQuantizer with trained entropy model
    """
    original_forward = module.forward
    
    def entropy_forward(x, *args, **kwargs):
        if hasattr(module, 'use_entropy_quant') and module.use_entropy_quant:
            # Get original weights
            org_weight = module.org_weight
            if not org_weight.is_cuda:
                org_weight = org_weight.cuda()
            
            # Quantize using entropy-aware quantizer
            q_weight = quantizer.quantize_weights(org_weight)
            
            # Apply quantized weights
            if hasattr(module, 'org_module'):
                orig_w = module.org_module.weight.data
                module.org_module.weight.data = q_weight
                
                # Apply activation quantization if enabled
                if hasattr(module, 'act_quantizer') and module.act_quantizer is not None:
                    x = module.act_quantizer(x)
                
                # Forward pass
                out = module.org_module(x, *args, **kwargs)
                
                # Restore original weights
                module.org_module.weight.data = orig_w
                return out
        
        return original_forward(x, *args, **kwargs)
    
    module.forward = entropy_forward


def save_entropy_models(cluster_quantizers: Dict, save_path: str):
    """
    Save trained entropy models
    
    Args:
        cluster_quantizers: Dictionary of quantizers
        save_path: Path to save
    """
    logger.info(f"Saving entropy models to {save_path}")
    
    entropy_states = {}
    for name, quantizer in cluster_quantizers.items():
        if quantizer.is_trained and quantizer.entropy_model is not None:
            entropy_states[name] = {
                'entropy_model': quantizer.entropy_model.state_dict(),
                'bit_allocation': quantizer.bit_allocation.cpu(),
                'num_clusters': quantizer.num_clusters
            }
    
    torch.save(entropy_states, save_path)
    logger.info(f"Saved {len(entropy_states)} entropy models")


def load_entropy_models(cluster_quantizers: Dict, load_path: str):
    """
    Load pre-trained entropy models
    
    Args:
        cluster_quantizers: Dictionary of quantizers
        load_path: Path to load from
    """
    from entropy_model import ClusterEntropyModel
    
    logger.info(f"Loading entropy models from {load_path}")
    entropy_states = torch.load(load_path)
    
    loaded_count = 0
    for name, quantizer in cluster_quantizers.items():
        if name in entropy_states:
            state = entropy_states[name]
            
            # Create and load entropy model
            quantizer.entropy_model = ClusterEntropyModel(
                num_clusters=state['num_clusters'],
                max_levels=256
            ).cuda()
            quantizer.entropy_model.load_state_dict(state['entropy_model'])
            quantizer.entropy_model.eval()
            
            # Load bit allocation
            quantizer.bit_allocation = state['bit_allocation'].cuda()
            quantizer.is_trained = True
            
            loaded_count += 1
    
    logger.info(f"Loaded {loaded_count} entropy models")