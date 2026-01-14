"""
Standalone test script for compression pipeline
Tests all components independently
"""

import torch
import numpy as np
import logging
from typing import Dict

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def test_huffman_encoder():
    """Test basic Huffman encoding/decoding"""
    from huffman_codec import HuffmanEncoder, BitPacker
    
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Huffman Encoder")
    logger.info("="*80)
    
    # Create test probability distribution
    probabilities = {
        0: 0.5,   # Very common
        1: 0.3,   # Common
        2: 0.15,  # Less common
        3: 0.05   # Rare
    }
    
    # Create encoder
    encoder = HuffmanEncoder(probabilities)
    
    logger.info("\nHuffman Codes:")
    for symbol, code in encoder.codes.items():
        logger.info(f"  Symbol {symbol}: {code} ({len(code)} bits)")
    
    # Calculate average code length
    avg_length = encoder.get_average_bitlength()
    
    # Calculate Shannon entropy
    shannon_entropy = -sum(p * np.log2(p) for p in probabilities.values())
    
    logger.info(f"\nShannon entropy: {shannon_entropy:.3f} bits")
    logger.info(f"Huffman avg length: {avg_length:.3f} bits")
    logger.info(f"Overhead: {(avg_length - shannon_entropy):.3f} bits ({(avg_length/shannon_entropy - 1)*100:.2f}%)")
    
    # Test encoding/decoding
    test_symbols = [0, 0, 1, 0, 2, 0, 1, 3, 0, 1]  # Sample sequence
    
    bitstring = encoder.encode(test_symbols)
    decoded = encoder.decode(bitstring)
    
    logger.info(f"\nOriginal symbols: {test_symbols}")
    logger.info(f"Bitstring: {bitstring}")
    logger.info(f"Decoded symbols: {decoded}")
    logger.info(f"Match: {test_symbols == decoded}")
    
    # Test bit packing
    packed_bytes, padding = BitPacker.pack(bitstring)
    unpacked = BitPacker.unpack(packed_bytes, padding)
    
    logger.info(f"\nBitstring length: {len(bitstring)} bits")
    logger.info(f"Packed size: {len(packed_bytes)} bytes")
    logger.info(f"Padding: {padding} bits")
    logger.info(f"Unpacked matches: {bitstring == unpacked}")
    
    logger.info("\n✓ Huffman encoder test PASSED")
    return True


def test_layer_compression():
    """Test compressing a single layer"""
    from huffman_codec import LayerCompressor, HuffmanEncoder
    from integrate_entropy_model import ClusterBasedQuantizer
    from entropy_model import ClusterEntropyModel
    
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Layer Compression")
    logger.info("="*80)
    
    # Create synthetic weight tensor
    torch.manual_seed(42)
    weights = torch.randn(256, 128).cuda()  # 32K weights
    
    logger.info(f"Test weight tensor: {weights.shape} ({weights.numel():,} weights)")
    
    # Create mock quantizer
    num_clusters = 4
    clusters = [torch.randn(10).cuda() for _ in range(num_clusters)]
    densities = [0.4, 0.3, 0.2, 0.1]
    assignments = torch.randint(0, num_clusters, weights.shape).cuda()
    
    quantizer = ClusterBasedQuantizer(
        clusters=clusters,
        densities=densities,
        assignments=assignments,
        num_clusters=num_clusters
    )
    
    # Train entropy model
    logger.info("\nTraining entropy model...")
    quantizer.entropy_model = ClusterEntropyModel(num_clusters, max_levels=256).cuda()
    quantizer.train_entropy_model(weights, num_iterations=100)
    
    # Compress layer
    logger.info("\nCompressing layer...")
    compressor = LayerCompressor(quantizer, cluster_id=0)
    compressed_data = compressor.compress(weights)
    
    # Decompress
    logger.info("\nDecompressing layer...")
    decompressed_weights = compressor.decompress(compressed_data)
    
    # Check reconstruction
    quantized_original = quantizer.quantize_weights(weights)
    mse = torch.mean((decompressed_weights.cuda() - quantized_original) ** 2).item()
    max_diff = torch.max(torch.abs(decompressed_weights.cuda() - quantized_original)).item()
    
    logger.info(f"\nReconstruction quality:")
    logger.info(f"  MSE: {mse:.6e}")
    logger.info(f"  Max difference: {max_diff:.6e}")
    logger.info(f"  Perfect reconstruction: {mse < 1e-6}")
    
    # Compression stats
    original_bits = weights.numel() * 32
    compressed_bits = len(compressed_data['compressed_bytes']) * 8
    compression_ratio = original_bits / compressed_bits
    
    logger.info(f"\nCompression statistics:")
    logger.info(f"  Original (FP32): {original_bits/8/1024:.2f} KB")
    logger.info(f"  Compressed: {compressed_bits/8/1024:.2f} KB")
    logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
    
    logger.info("\n✓ Layer compression test PASSED")
    return True


def test_full_model_compression():
    """Test compressing multiple layers like a real model"""
    from huffman_codec import ModelCompressor, LayerCompressor
    from integrate_entropy_model import ClusterBasedQuantizer
    from entropy_model import ClusterEntropyModel
    
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Full Model Compression")
    logger.info("="*80)
    
    # Create mock model with multiple layers
    class MockModule:
        def __init__(self, shape):
            self.org_weight = torch.randn(shape).cuda()
            self.weight = torch.nn.Parameter(self.org_weight.clone())
    
    class MockModel:
        def __init__(self):
            self.layer1 = MockModule((512, 256))
            self.layer2 = MockModule((256, 128))
            self.layer3 = MockModule((128, 64))
        
        def named_modules(self):
            return [
                ('layer1', self.layer1),
                ('layer2', self.layer2),
                ('layer3', self.layer3)
            ]
    
    model = MockModel()
    
    # Create quantizers for each layer
    cluster_quantizers = {}
    for name, module in model.named_modules():
        num_clusters = 4
        shape = module.org_weight.shape
        
        clusters = [torch.randn(10).cuda() for _ in range(num_clusters)]
        densities = [0.4, 0.3, 0.2, 0.1]
        assignments = torch.randint(0, num_clusters, shape).cuda()
        
        quantizer = ClusterBasedQuantizer(
            clusters=clusters,
            densities=densities,
            assignments=assignments,
            num_clusters=num_clusters
        )
        
        # Train entropy model
        quantizer.entropy_model = ClusterEntropyModel(num_clusters, max_levels=256).cuda()
        quantizer.train_entropy_model(module.org_weight, num_iterations=50)
        
        cluster_quantizers[name] = quantizer
    
    # Compress entire model
    logger.info("\nCompressing all layers...")
    model_compressor = ModelCompressor(cluster_quantizers)
    compressed_layers = model_compressor.compress_model(model)
    
    # Test saving/loading
    logger.info("\nTesting save/load...")
    save_path = "/tmp/test_compressed_model.pkl"
    model_compressor.save_compressed_model(save_path)
    
    loaded_compressed, loaded_configs = ModelCompressor.load_compressed_model(save_path)
    
    logger.info(f"  Saved layers: {len(compressed_layers)}")
    logger.info(f"  Loaded layers: {len(loaded_compressed)}")
    logger.info(f"  Match: {len(compressed_layers) == len(loaded_compressed)}")
    
    # Test decompression
    logger.info("\nTesting decompression...")
    model_compressor.compressed_layers = loaded_compressed
    
    for name in compressed_layers.keys():
        original = None
        for module_name, module in model.named_modules():
            if module_name == name:
                original = module.org_weight
                break
        
        if original is not None:
            quantized = cluster_quantizers[name].quantize_weights(original)
            decompressed = model_compressor.decompress_layer(name)
            
            mse = torch.mean((decompressed.cuda() - quantized) ** 2).item()
            logger.info(f"  {name}: MSE = {mse:.6e}")
    
    logger.info("\n✓ Full model compression test PASSED")
    return True


def test_compression_vs_theory():
    """Compare theoretical Shannon entropy vs actual Huffman compression"""
    from huffman_codec import HuffmanEncoder
    
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Shannon Entropy vs Huffman Compression")
    logger.info("="*80)
    
    test_cases = [
        {
            'name': 'Uniform distribution',
            'probs': {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        },
        {
            'name': 'Skewed distribution',
            'probs': {0: 0.7, 1: 0.2, 2: 0.08, 3: 0.02}
        },
        {
            'name': 'Very skewed',
            'probs': {0: 0.9, 1: 0.06, 2: 0.03, 3: 0.01}
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n{test_case['name']}:")
        probs = test_case['probs']
        
        # Shannon entropy
        shannon = -sum(p * np.log2(p) for p in probs.values() if p > 0)
        
        # Huffman
        encoder = HuffmanEncoder(probs)
        huffman_avg = encoder.get_average_bitlength()
        
        # Overhead
        overhead = huffman_avg - shannon
        overhead_pct = (overhead / shannon) * 100
        
        logger.info(f"  Shannon entropy: {shannon:.4f} bits")
        logger.info(f"  Huffman average: {huffman_avg:.4f} bits")
        logger.info(f"  Overhead: {overhead:.4f} bits ({overhead_pct:.2f}%)")
        
        # Generate sample and test
        np.random.seed(42)
        symbols = np.random.choice(
            list(probs.keys()),
            size=10000,
            p=list(probs.values())
        )
        
        bitstring = encoder.encode(symbols.tolist())
        actual_avg = len(bitstring) / len(symbols)
        
        logger.info(f"  Actual (10K samples): {actual_avg:.4f} bits")
        logger.info(f"  Matches Huffman: {abs(actual_avg - huffman_avg) < 0.01}")
    
    logger.info("\n✓ Theory vs practice test PASSED")
    return True


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("RUNNING ALL COMPRESSION TESTS")
    logger.info("="*80)
    
    tests = [
        ("Huffman Encoder", test_huffman_encoder),
        ("Layer Compression", test_layer_compression),
        ("Full Model Compression", test_full_model_compression),
        ("Theory vs Practice", test_compression_vs_theory)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"\n❌ {test_name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
    else:
        logger.info("\n⚠️  SOME TESTS FAILED")
    
    logger.info("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    import sys
    
    # Check if specific test requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        tests = {
            'huffman': test_huffman_encoder,
            'layer': test_layer_compression,
            'model': test_full_model_compression,
            'theory': test_compression_vs_theory
        }
        
        if test_name in tests:
            tests[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {list(tests.keys())}")
    else:
        # Run all tests
        run_all_tests()