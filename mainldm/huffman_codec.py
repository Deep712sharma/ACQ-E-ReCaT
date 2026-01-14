"""
Huffman Coding Implementation for Weight Compression
Provides actual encoding/decoding using learned probability distributions
"""

import torch
import numpy as np
import pickle
import heapq
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HuffmanNode:
    """Node for Huffman tree"""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanEncoder:
    """
    Huffman encoder that builds codes from probability distribution
    """
    def __init__(self, probabilities: Dict[int, float]):
        """
        Args:
            probabilities: Dict mapping symbol (int) to probability (float)
        """
        self.probabilities = probabilities
        self.codes = {}
        self.reverse_codes = {}
        self._build_codes()
    
    def _build_codes(self):
        """Build Huffman codes from probability distribution"""
        if len(self.probabilities) == 1:
            # Special case: only one symbol
            symbol = list(self.probabilities.keys())[0]
            self.codes[symbol] = '0'
            self.reverse_codes['0'] = symbol
            return
        
        # Create priority queue (min heap)
        heap = []
        for symbol, prob in self.probabilities.items():
            node = HuffmanNode(symbol=symbol, freq=prob)
            heapq.heappush(heap, node)
        
        # Build Huffman tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            parent = HuffmanNode(
                freq=left.freq + right.freq,
                left=left,
                right=right
            )
            heapq.heappush(heap, parent)
        
        # Extract codes by traversing tree
        root = heap[0]
        self._extract_codes(root, "")
        
        # Build reverse mapping for decoding
        self.reverse_codes = {code: symbol for symbol, code in self.codes.items()}
    
    def _extract_codes(self, node: HuffmanNode, code: str):
        """Recursively extract codes from Huffman tree"""
        if node.symbol is not None:
            # Leaf node
            self.codes[node.symbol] = code if code else '0'
            return
        
        if node.left:
            self._extract_codes(node.left, code + '0')
        if node.right:
            self._extract_codes(node.right, code + '1')
    
    def encode(self, symbols: List[int]) -> str:
        """
        Encode list of symbols to bitstring
        
        Args:
            symbols: List of integer symbols
            
        Returns:
            bitstring: String of '0's and '1's
        """
        return ''.join(self.codes[s] for s in symbols)
    
    def decode(self, bitstring: str) -> List[int]:
        """
        Decode bitstring back to symbols
        
        Args:
            bitstring: String of '0's and '1's
            
        Returns:
            symbols: List of decoded integer symbols
        """
        symbols = []
        current_code = ""
        
        for bit in bitstring:
            current_code += bit
            if current_code in self.reverse_codes:
                symbols.append(self.reverse_codes[current_code])
                current_code = ""
        
        if current_code:
            logger.warning(f"Incomplete code at end: {current_code}")
        
        return symbols
    
    def get_average_bitlength(self) -> float:
        """Calculate average code length (should equal entropy)"""
        return sum(self.probabilities[s] * len(self.codes[s]) 
                   for s in self.probabilities.keys())


class BitPacker:
    """
    Pack bitstrings into bytes for efficient storage
    """
    @staticmethod
    def pack(bitstring: str) -> Tuple[bytes, int]:
        """
        Pack bitstring into bytes
        
        Args:
            bitstring: String of '0's and '1's
            
        Returns:
            packed_bytes: Packed bytes
            num_padding_bits: Number of padding bits added
        """
        # Calculate padding needed
        num_bits = len(bitstring)
        num_padding_bits = (8 - num_bits % 8) % 8
        
        # Add padding
        padded = bitstring + '0' * num_padding_bits
        
        # Convert to bytes
        packed_bytes = int(padded, 2).to_bytes(len(padded) // 8, byteorder='big')
        
        return packed_bytes, num_padding_bits
    
    @staticmethod
    def unpack(packed_bytes: bytes, num_padding_bits: int) -> str:
        """
        Unpack bytes back to bitstring
        
        Args:
            packed_bytes: Packed bytes
            num_padding_bits: Number of padding bits to remove
            
        Returns:
            bitstring: Original bitstring
        """
        # Convert bytes to bitstring
        num_bits = len(packed_bytes) * 8
        bitstring = bin(int.from_bytes(packed_bytes, byteorder='big'))[2:].zfill(num_bits)
        
        # Remove padding
        if num_padding_bits > 0:
            bitstring = bitstring[:-num_padding_bits]
        
        return bitstring


class LayerCompressor:
    """
    Compress a single layer using Huffman coding with cluster-based quantization
    """
    def __init__(self, quantizer, cluster_id: int = 0):
        """
        Args:
            quantizer: ClusterBasedQuantizer instance
            cluster_id: Which cluster to compress
        """
        self.quantizer = quantizer
        self.cluster_id = cluster_id
        self.encoder = None
        self.metadata = {}
    
    def compress(self, weights: torch.Tensor) -> Dict:
        """
        Compress weights using learned probability distribution
        
        Args:
            weights: Weight tensor to compress
            
        Returns:
            compressed_data: Dictionary with compressed representation
        """
        # Quantize weights
        quantized_weights = self.quantizer.quantize_weights(weights)
        
        # Get bit allocation for this cluster
        n_bits = int(self.quantizer.bit_allocation[self.cluster_id].item())
        n_levels = 2 ** n_bits
        
        # Get learned probability distribution
        if self.quantizer.entropy_model is not None:
            probs = self.quantizer.entropy_model.get_cluster_probs(self.cluster_id)
            probs = probs[:n_levels]
            probs = probs / probs.sum()  # Normalize
            probs = probs.detach().cpu().numpy()
        else:
            # Fallback: uniform distribution
            probs = np.ones(n_levels) / n_levels
        
        # Convert weights to discrete indices
        w_min = quantized_weights.min().item()
        w_max = quantized_weights.max().item()
        
        if abs(w_max - w_min) < 1e-8:
            # All weights are the same
            symbols = [0] * quantized_weights.numel()
        else:
            normalized = (quantized_weights - w_min) / (w_max - w_min)
            indices = (normalized * (n_levels - 1)).round().long()
            symbols = indices.flatten().cpu().tolist()
        
        # Build Huffman encoder from learned probabilities
        prob_dict = {i: float(probs[i]) for i in range(n_levels)}
        self.encoder = HuffmanEncoder(prob_dict)
        
        # Encode to bitstring
        bitstring = self.encoder.encode(symbols)
        
        # Pack into bytes
        packed_bytes, num_padding = BitPacker.pack(bitstring)
        
        # Store metadata
        self.metadata = {
            'shape': weights.shape,
            'w_min': w_min,
            'w_max': w_max,
            'n_levels': n_levels,
            'n_bits': n_bits,
            'num_symbols': len(symbols),
            'num_padding_bits': num_padding,
            'huffman_codes': self.encoder.codes,
            'probabilities': prob_dict
        }
        
        # Calculate compression stats
        original_bits = weights.numel() * 32  # FP32
        nominal_bits = weights.numel() * n_bits
        compressed_bits = len(packed_bytes) * 8
        
        logger.info(f"  Compression stats:")
        logger.info(f"    Original (FP32): {original_bits} bits ({original_bits/8/1024:.2f} KB)")
        logger.info(f"    Nominal ({n_bits}-bit): {nominal_bits} bits ({nominal_bits/8/1024:.2f} KB)")
        logger.info(f"    Compressed (Huffman): {compressed_bits} bits ({compressed_bits/8/1024:.2f} KB)")
        logger.info(f"    Ratio vs FP32: {original_bits/compressed_bits:.2f}x")
        logger.info(f"    Ratio vs nominal: {nominal_bits/compressed_bits:.2f}x")
        logger.info(f"    Actual bits/weight: {compressed_bits/weights.numel():.3f}")
        
        return {
            'compressed_bytes': packed_bytes,
            'metadata': self.metadata,
            'encoder': self.encoder
        }
    
    def decompress(self, compressed_data: Dict) -> torch.Tensor:
        """
        Decompress weights back to tensor
        
        Args:
            compressed_data: Dictionary from compress()
            
        Returns:
            weights: Decompressed weight tensor
        """
        packed_bytes = compressed_data['compressed_bytes']
        metadata = compressed_data['metadata']
        encoder = compressed_data['encoder']
        
        # Unpack bytes to bitstring
        bitstring = BitPacker.unpack(packed_bytes, metadata['num_padding_bits'])
        
        # Decode bitstring to symbols
        symbols = encoder.decode(bitstring)
        
        # Verify length
        if len(symbols) != metadata['num_symbols']:
            logger.warning(f"Symbol count mismatch: {len(symbols)} vs {metadata['num_symbols']}")
            # Pad or truncate
            if len(symbols) < metadata['num_symbols']:
                symbols.extend([0] * (metadata['num_symbols'] - len(symbols)))
            else:
                symbols = symbols[:metadata['num_symbols']]
        
        # Convert symbols back to weights
        indices_tensor = torch.tensor(symbols, dtype=torch.float32)
        normalized = indices_tensor / (metadata['n_levels'] - 1)
        weights = normalized * (metadata['w_max'] - metadata['w_min']) + metadata['w_min']
        
        # Reshape to original shape
        weights = weights.reshape(metadata['shape'])
        
        return weights


class ModelCompressor:
    """
    Compress entire model using cluster-based quantization + Huffman coding
    """
    def __init__(self, cluster_quantizers: Dict):
        """
        Args:
            cluster_quantizers: Dict of layer_name -> ClusterBasedQuantizer
        """
        self.cluster_quantizers = cluster_quantizers
        self.compressed_layers = {}
    
    def compress_model(self, model) -> Dict:
        """
        Compress all quantizable layers in model
        
        Args:
            model: PyTorch model with org_weight attributes
            
        Returns:
            compressed_model: Dictionary with all compressed layers
        """
        logger.info("\n" + "="*80)
        logger.info("COMPRESSING MODEL WITH HUFFMAN CODING")
        logger.info("="*80)
        
        total_original_bits = 0
        total_compressed_bits = 0
        
        for name, module in model.named_modules():
            if not hasattr(module, 'org_weight'):
                continue
            
            if name not in self.cluster_quantizers:
                continue
            
            logger.info(f"\nCompressing layer: {name}")
            
            quantizer = self.cluster_quantizers[name]
            org_weight = module.org_weight.cuda()
            
            # Compress layer
            compressor = LayerCompressor(quantizer, cluster_id=0)
            compressed_data = compressor.compress(org_weight)
            
            self.compressed_layers[name] = compressed_data
            
            # Accumulate stats
            total_original_bits += org_weight.numel() * 32
            total_compressed_bits += len(compressed_data['compressed_bytes']) * 8
        
        # Overall statistics
        logger.info("\n" + "="*80)
        logger.info("COMPRESSION SUMMARY")
        logger.info(f"  Layers compressed: {len(self.compressed_layers)}")
        logger.info(f"  Total original size: {total_original_bits/8/1024/1024:.2f} MB")
        logger.info(f"  Total compressed size: {total_compressed_bits/8/1024/1024:.2f} MB")
        logger.info(f"  Overall compression ratio: {total_original_bits/total_compressed_bits:.2f}x")
        logger.info("="*80 + "\n")
        
        return self.compressed_layers
    
    def decompress_layer(self, layer_name: str) -> torch.Tensor:
        """
        Decompress a specific layer
        
        Args:
            layer_name: Name of layer to decompress
            
        Returns:
            weights: Decompressed weights
        """
        if layer_name not in self.compressed_layers:
            raise ValueError(f"Layer {layer_name} not found in compressed layers")
        
        compressed_data = self.compressed_layers[layer_name]
        quantizer = self.cluster_quantizers[layer_name]
        
        compressor = LayerCompressor(quantizer, cluster_id=0)
        weights = compressor.decompress(compressed_data)
        
        return weights
    
    def save_compressed_model(self, save_path: str):
        """
        Save compressed model to disk
        
        Args:
            save_path: Path to save compressed model
        """
        logger.info(f"Saving compressed model to {save_path}")
        
        save_data = {
            'compressed_layers': self.compressed_layers,
            'quantizer_configs': {}
        }
        
        # Save quantizer configurations (needed for decompression)
        for name, quantizer in self.cluster_quantizers.items():
            save_data['quantizer_configs'][name] = {
                'num_clusters': quantizer.num_clusters,
                'bit_allocation': quantizer.bit_allocation.cpu(),
                'clusters': quantizer.clusters,
                'densities': quantizer.densities,
                'assignments': quantizer.assignments
            }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Calculate file size
        import os
        file_size_mb = os.path.getsize(save_path) / 1024 / 1024
        logger.info(f"Compressed model saved: {file_size_mb:.2f} MB")
    
    @staticmethod
    def load_compressed_model(load_path: str) -> Tuple[Dict, Dict]:
        """
        Load compressed model from disk
        
        Args:
            load_path: Path to compressed model
            
        Returns:
            compressed_layers: Compressed layer data
            quantizer_configs: Quantizer configurations
        """
        logger.info(f"Loading compressed model from {load_path}")
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        return save_data['compressed_layers'], save_data['quantizer_configs']


def create_decompression_wrapper(model, compressed_layers: Dict, cluster_quantizers: Dict):
    """
    Wrap model to decompress weights on-the-fly during forward pass
    
    Args:
        model: PyTorch model
        compressed_layers: Compressed layer data
        cluster_quantizers: Quantizer instances
    """
    compressor = ModelCompressor(cluster_quantizers)
    compressor.compressed_layers = compressed_layers
    
    for name, module in model.named_modules():
        if name not in compressed_layers:
            continue
        
        # Store original forward
        original_forward = module.forward
        
        # Create decompression wrapper
        def make_forward(module_name, orig_forward):
            def decompressing_forward(self, x, *args, **kwargs):
                # Decompress weights
                weights = compressor.decompress_layer(module_name)
                
                # Temporarily set weights
                if hasattr(self, 'org_module'):
                    old_weight = self.org_module.weight.data
                    self.org_module.weight.data = weights.cuda()
                    result = orig_forward(x, *args, **kwargs)
                    self.org_module.weight.data = old_weight
                else:
                    old_weight = self.weight.data
                    self.weight.data = weights.cuda()
                    result = orig_forward(x, *args, **kwargs)
                    self.weight.data = old_weight
                
                return result
            return decompressing_forward
        
        # Replace forward
        module.forward = make_forward(name, original_forward).__get__(module, type(module))
        
        logger.info(f"Wrapped {name} with decompression")