"""
Learned Entropy Model for Neural Network Quantization
Estimates probability distributions of quantized weights for better compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ClusterEntropyModel(nn.Module):
    """
    Entropy model specifically designed for cluster-based quantization.
    Learns probability distribution for each cluster's quantized values.
    """
    def __init__(self, num_clusters: int, max_levels: int = 256):
        super().__init__()
        self.num_clusters = num_clusters
        self.max_levels = max_levels
        
        # Learned probability mass function for each cluster
        # Shape: [num_clusters, max_levels]
        self.cluster_logits = nn.Parameter(torch.randn(num_clusters, max_levels))
        
        # Per-cluster scale parameters (learnable)
        self.cluster_scales = nn.Parameter(torch.ones(num_clusters))
        
    def get_cluster_probs(self, cluster_id: int) -> torch.Tensor:
        """Get probability distribution for a specific cluster"""
        logits = self.cluster_logits[cluster_id]
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def estimate_entropy_for_cluster(self, 
                                     quantized_weights: torch.Tensor,
                                     cluster_id: int,
                                     n_levels: int) -> torch.Tensor:
        """
        Estimate entropy for weights belonging to a specific cluster
        
        Args:
            quantized_weights: Quantized weights in this cluster
            cluster_id: ID of the cluster
            n_levels: Number of quantization levels used for this cluster
            
        Returns:
            bits_per_weight: Estimated bits per weight
        """
        # Get probability distribution for this cluster
        probs = self.get_cluster_probs(cluster_id)
        
        # Use only the relevant levels based on n_levels
        probs = probs[:n_levels]
        probs = probs / probs.sum()  # Renormalize
        
        # Convert weights to quantization indices
        w_min = quantized_weights.min()
        w_max = quantized_weights.max()
        
        if w_max - w_min < 1e-8:
            return torch.tensor(0.0, device=quantized_weights.device)
        
        # Normalize to [0, n_levels-1]
        normalized = (quantized_weights - w_min) / (w_max - w_min)
        soft_indices = normalized * (n_levels - 1)
        soft_indices = torch.clamp(soft_indices, 0, n_levels - 1)
        
        # Soft indexing for differentiability
        indices_floor = soft_indices.long()
        indices_ceil = torch.clamp(indices_floor + 1, max=n_levels - 1)
        alpha = soft_indices - indices_floor.float()
        
        # Interpolate probabilities
        prob_floor = probs[indices_floor]
        prob_ceil = probs[indices_ceil]
        interpolated_probs = (1 - alpha) * prob_floor + alpha * prob_ceil
        
        # Calculate bits: -log2(p)
        bits_per_weight = -torch.log2(interpolated_probs + 1e-10)
        
        return bits_per_weight.mean()
    
    def estimate_total_entropy(self,
                              quantized_weights: torch.Tensor,
                              cluster_assignments: torch.Tensor,
                              bit_allocation: torch.Tensor) -> torch.Tensor:
        """
        Estimate total entropy across all clusters
        
        Args:
            quantized_weights: All quantized weights
            cluster_assignments: Cluster assignment for each weight
            bit_allocation: Bits allocated to each cluster
            
        Returns:
            total_bits: Total estimated bits
        """
        weights_flat = quantized_weights.flatten()
        assignments_flat = cluster_assignments.flatten()
        
        total_bits = torch.tensor(0.0, device=quantized_weights.device)
        total_weights = 0
        
        for cluster_id in range(self.num_clusters):
            mask = (assignments_flat == cluster_id)
            cluster_weights = weights_flat[mask]
            
            if cluster_weights.numel() == 0:
                continue
            
            n_levels = 2 ** int(bit_allocation[cluster_id].item())
            cluster_bits = self.estimate_entropy_for_cluster(
                cluster_weights, 
                cluster_id, 
                n_levels
            )
            
            total_bits += cluster_bits * cluster_weights.numel()
            total_weights += cluster_weights.numel()
        
        if total_weights > 0:
            return total_bits / total_weights
        return torch.tensor(0.0, device=quantized_weights.device)


class EntropyModelTrainer:
    """
    Trainer for cluster-based entropy model
    """
    def __init__(self, 
                 entropy_model: ClusterEntropyModel,
                 lr: float = 1e-3):
        self.entropy_model = entropy_model
        self.optimizer = torch.optim.Adam(entropy_model.parameters(), lr=lr)
        
    def train_step(self, 
                   quantized_weights: torch.Tensor,
                   cluster_assignments: torch.Tensor,
                   bit_allocation: torch.Tensor) -> Dict:
        """
        Single training step
        
        Args:
            quantized_weights: Quantized weights (detached)
            cluster_assignments: Cluster assignments
            bit_allocation: Bits per cluster
            
        Returns:
            metrics: Training metrics
        """
        self.optimizer.zero_grad()
        
        # Ensure no gradients from quantization
        quantized_weights = quantized_weights.detach()
        
        # Estimate entropy
        avg_bits = self.entropy_model.estimate_total_entropy(
            quantized_weights,
            cluster_assignments,
            bit_allocation
        )
        
        # Loss: minimize negative log-likelihood (bits)
        loss = avg_bits
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'bits_per_weight': loss.item()
        }
    
    def train(self,
              quantized_weights: torch.Tensor,
              cluster_assignments: torch.Tensor,
              bit_allocation: torch.Tensor,
              num_iterations: int = 500) -> Dict:
        """
        Train entropy model
        
        Args:
            quantized_weights: Quantized weights
            cluster_assignments: Cluster assignments
            bit_allocation: Bits per cluster
            num_iterations: Training iterations
            
        Returns:
            final_metrics: Final metrics
        """
        self.entropy_model.train()
        
        best_loss = float('inf')
        
        for i in range(num_iterations):
            metrics = self.train_step(
                quantized_weights,
                cluster_assignments,
                bit_allocation
            )
            
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Entropy training iter {i+1}/{num_iterations}: "
                          f"Bits/weight={metrics['bits_per_weight']:.3f}")
        
        self.entropy_model.eval()
        return {'bits_per_weight': best_loss}


def calculate_actual_bitrate(quantized_weights: torch.Tensor,
                             cluster_assignments: torch.Tensor,
                             bit_allocation: torch.Tensor,
                             entropy_model: ClusterEntropyModel) -> float:
    """
    Calculate actual bitrate using trained entropy model
    
    Args:
        quantized_weights: Quantized weight tensor
        cluster_assignments: Cluster assignments
        bit_allocation: Bits per cluster
        entropy_model: Trained entropy model
        
    Returns:
        bitrate: Actual bits per weight
    """
    entropy_model.eval()
    with torch.no_grad():
        estimated_bits = entropy_model.estimate_total_entropy(
            quantized_weights,
            cluster_assignments,
            bit_allocation
        )
    return estimated_bits.item()