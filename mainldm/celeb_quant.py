import sys
from load_clusters import prepare_cluster_data_for_quantization
from integrate_entropy_model import (
    ClusterBasedQuantizer, 
    calculate_average_bitwidth,
    find_matching_layer
)
# ADD NEW COMPRESSION IMPORTS
from compression_integration import (
    apply_entropy_aware_quantization_with_compression,
    benchmark_compression_decompression,
    compare_compression_methods,
    load_and_use_compressed_model
)
from huffman_codec import ModelCompressor

sys.path.append("./mainldm")
sys.path.append("./mainddpm")
sys.path.append('./src/taming-transformers')
sys.path.append('.')
print(sys.path)
import argparse
import os, gc
import time
import logging
import wandb
import numpy as np
import torch.distributed as dist

import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_trainer
from PIL import Image
from einops import rearrange
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from quant.utils import AttentionMap, seed_everything, Fisher 
from quant.quant_model import QModel
from quant.quant_block import Change_LDM_model_attnblock
from quant.set_quantize_params import set_act_quantize_params, set_weight_quantize_params
from quant.recon_Qmodel import recon_Qmodel, skip_LDM_Model
from quant.quant_layer import QuantModule
from average import average

logger = logging.getLogger(__name__)

def compute_actual_model_size(
    model,
    cluster_quantizers=None,
    fp32_bits=32
):
    total_bits = 0
    total_fp32_bits = 0

    for name, module in model.named_modules():
        if not hasattr(module, 'org_weight'):
            continue

        num_params = module.org_weight.numel()
        total_fp32_bits += num_params * fp32_bits

        # ── Quantized layer ─────────────────────────────
        if cluster_quantizers is not None and name in cluster_quantizers:
            quantizer = cluster_quantizers[name]

            # (1) assignment bits
            bits_per_weight = quantizer.get_average_bitwidth()
            total_bits += num_params * bits_per_weight

            # (2) codebook
            total_bits += quantizer.num_clusters * fp32_bits

            # (3) entropy model (OPTIONAL)
            entropy_model = getattr(quantizer, "entropy_model", None)
            if entropy_model is not None:
                for p in entropy_model.parameters():
                    total_bits += p.numel() * fp32_bits

        # ── Unquantized layer ────────────────────────────
        else:
            total_bits += num_params * fp32_bits

    total_MB = total_bits / 8 / (1024 ** 2)
    compression_ratio = total_fp32_bits / total_bits

    return total_bits, total_MB, compression_ratio


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0):
    ddim = DDIMSampler(model, slow_steps=model.interval_seq)
    ddim.quant_sample = True
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, dpm=False):
    log = dict()
    shape = [batch_size,
            model.model.diffusion_model.model.in_channels,
            model.model.diffusion_model.model.image_size,
            model.model.diffusion_model.model.image_size]

    with torch.no_grad():
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                            make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)
        t1 = time.time()
        x_sample = model.decode_first_stage(sample)
    torch.cuda.empty_cache()
    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    return log


def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
    n_samples=50000, dpm=False):

    tstart = time.time()
    n_saved = 0
    if model.cond_stage_model is None:
        all_images = []
        print(f"Running unconditional sampling for {n_samples} samples")
        with torch.no_grad():
            for _ in tqdm(range(n_samples // batch_size), desc="Sampling Batches (unconditional)"):
                logs = make_convolutional_sample(model, batch_size=batch_size,
                                                vanilla=vanilla, custom_steps=custom_steps,
                                                eta=eta, dpm=dpm)
                n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
                torch.cuda.empty_cache()

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("./mainldm/models/ldm/celeba256/config.yaml")  
    model = load_model_from_config(config, "./mainldm/models/ldm/celeba256/model.ckpt")
    return model


def load_cluster_data(clustering_dir, model):
    cluster_data = {
        'layer_clusters': {},
        'avg_bitwidth': 0,
        'total_params': 0
    }
    
    total_bits = 0
    total_params = 0
    
    logger.info(f"Loading cluster data from: {clustering_dir}")
    
    if not os.path.exists(clustering_dir):
        logger.error(f"Clustering directory does not exist: {clustering_dir}")
        logger.error("Please run the clustering script first!")
        return cluster_data
    
    cluster_files = [f for f in os.listdir(clustering_dir) if f.endswith('_clusters.pth')]
    
    if len(cluster_files) == 0:
        logger.error(f"No cluster files found in {clustering_dir}")
        logger.error("Please run the clustering script first!")
        return cluster_data
    
    logger.info(f"Found {len(cluster_files)} cluster files")
    
    for cluster_file in cluster_files:
        layer_name = cluster_file.replace('_clusters.pth', '')
        
        file_path = os.path.join(clustering_dir, cluster_file)
        try:
            cluster_info = torch.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load {cluster_file}: {e}")
            continue
        cluster_data['layer_clusters'][layer_name] = {
            'clusters': cluster_info['clusters'],
            'densities': cluster_info['densities'],
            'assignments': cluster_info['assignments']
        }
        
        logger.info(f"Loaded {layer_name}: {len(cluster_info['clusters'])} clusters")
    
    logger.info(f"\nTotal layers with clusters: {len(cluster_data['layer_clusters'])}\n")
    
    return cluster_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=30000)
    parser.add_argument('--sample_batch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument('--ddim_steps', type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1234+9)

    parser.add_argument("--replicate_interval", type=int, default=2)
    parser.add_argument("--sm_abit",type=int, default=8)
    parser.add_argument("--quant_act", action="store_true", default=True)
    parser.add_argument("--weight_bit",type=int, default=4)
    parser.add_argument("--act_bit",type=int,default=8)
    parser.add_argument("--quant_mode", type=str, default="qdiff", choices=["qdiff"])
    parser.add_argument("--split", action="store_true", default=True)
    parser.add_argument("--ptq", action="store_true", default=True)
    
    # Cluster-based quantization arguments
    parser.add_argument("--use_cluster_quant", action="store_true", default=True,
                    help="Use cluster-based mixed precision quantization")
    parser.add_argument("--clustering_dir", type=str, default="./clustering_per_layer",
                    help="Directory with pre-computed clustering files")
    
    # Entropy model arguments
    parser.add_argument("--use_entropy_model", action="store_true", default=False,
                       help="Use learned entropy model for bit allocation")
    parser.add_argument("--train_entropy", action="store_true", default=False,
                       help="Train entropy models (vs loading pretrained)")
    parser.add_argument("--entropy_iters", type=int, default=500,
                       help="Number of iterations to train entropy model")
    parser.add_argument("--entropy_models_path", type=str, default="./entropy_models.pth",
                       help="Path to save/load entropy models")
    
    # NEW: Compression arguments
    parser.add_argument("--use_compression", action="store_true", default=False,
                       help="Actually compress weights (vs just estimate)")
    parser.add_argument("--compressed_model_path", type=str, 
                       default="./compressed_celeb_model.pkl",
                       help="Path to save/load compressed model")
    parser.add_argument("--benchmark_compression", action="store_true", default=False,
                       help="Benchmark compression/decompression speed")
    parser.add_argument("--compare_methods", action="store_true", default=False,
                       help="Compare Shannon entropy vs Huffman coding")
    parser.add_argument("--load_compressed", action="store_true", default=False,
                       help="Load pre-compressed model instead of compressing")
    
    args = parser.parse_args()
    args.mode = "dps_opt"

    seed_everything(args.seed)
    device = torch.device("cuda", args.local_rank)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("./run.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logging.info(args)
    
    logger.info("load calibration...")
    interval_seq, all_cali_data, all_t, all_cali_t, all_cache = \
            torch.load("./calibration/celeba{}_cache{}_{}.pth".format(args.ddim_steps, args.replicate_interval, args.mode))
    logger.info("load calibration down!")
    args.interval_seq = interval_seq
    logger.info(f"The interval_seq: {args.interval_seq}")
    
    model = get_model()
    model.model_ema.store(model.model.parameters())
    model.model_ema.copy_to(model.model)

    (a_list, b_list) = torch.load(f"./error_dec/celeb/pre_cacheerr_abCov_interval{args.replicate_interval}_list.pth")
    model.model.diffusion_model.a_list = torch.stack(a_list)
    model.model.diffusion_model.b_list = torch.stack(b_list)
    model.model.diffusion_model.timesteps = args.ddim_steps

    if args.ptq:
        wq_params = {'n_bits': args.weight_bit, 'symmetric': False, 'channel_wise': True, 'scale_method': 'mse'}
        aq_params = {'n_bits': args.act_bit, 'symmetric': False, 'channel_wise': False, 'scale_method': 'mse', 
                    'leaf_param': args.quant_act, "prob": 1.0, "num_timesteps": args.ddim_steps}
        
        q_unet = QModel(model.model.diffusion_model, args, wq_params=wq_params, aq_params=aq_params)
        q_unet.cuda()
        q_unet.eval()

        logger.info("Setting the first and the last layer to 8-bit")
        q_unet.set_first_last_layer_to_8bit()
        q_unet.set_quant_state(False, False)

        if args.split:
            q_unet.model.split_shortcut = True

        cali_data = torch.cat(all_cali_data)
        t = torch.cat(all_t)
        idx = torch.randperm(len(cali_data))[:8]
        cali_data = cali_data[idx]
        t = t[idx]

        logger.info("Setting activation quantization parameters...")
        set_act_quantize_params(args.interval_seq, q_unet, all_cali_data, all_t, all_cache)

        bitrate_info = None
        
        # CLUSTER-BASED QUANTIZATION WITH ENTROPY MODEL AND COMPRESSION
        if args.use_cluster_quant and args.use_entropy_model:
            
            # Load cluster data
            cluster_data = load_cluster_data(args.clustering_dir, q_unet)
            
            # OPTION 1: Load pre-compressed model
            if args.load_compressed:
                logger.info("\n📦 Loading pre-compressed model...")
                model_compressor = load_and_use_compressed_model(
                    args.compressed_model_path,
                    q_unet,
                    test_inference=False
                )
                
                # Get quantizers and compressed layers
                compressed_layers, quantizer_configs = ModelCompressor.load_compressed_model(
                    args.compressed_model_path
                )
                
                # Recreate quantizers
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
                
                # Decompress weights into model
                model_compressor = ModelCompressor(cluster_quantizers)
                model_compressor.compressed_layers = compressed_layers
                
                logger.info("🔄 Decompressing weights into model...")
                for name in compressed_layers.keys():
                    weights = model_compressor.decompress_layer(name)
                    for module_name, module in q_unet.named_modules():
                        if module_name == name and hasattr(module, 'org_weight'):
                            if hasattr(module, 'org_module'):
                                module.org_module.weight.data = weights.cuda()
                            elif hasattr(module, 'weight'):
                                module.weight.data = weights.cuda()
                            logger.info(f"  ✓ Loaded: {name}")
                
                logger.info("✅ Compressed model loaded and decompressed")
                bitrate_info = None
            
            # OPTION 2: Compress model now
            else:
                logger.info("\n🗜️  Compressing model with Huffman coding...")
                
                # Apply compression-aware quantization
                cluster_quantizers, bitrate_info, compressed_layers = \
                    apply_entropy_aware_quantization_with_compression(
                        q_unet,
                        cluster_data,
                        args,
                        train_entropy=args.train_entropy,
                        num_entropy_iters=args.entropy_iters,
                        use_compression=args.use_compression,
                        save_compressed=True,
                        compressed_model_path=args.compressed_model_path
                    )
                
                # Optional: Benchmark performance
                if args.benchmark_compression and compressed_layers:
                    benchmark_compression_decompression(
                        compressed_layers,
                        cluster_quantizers,
                        num_trials=10
                    )
                
                # Optional: Compare theory vs practice
                if args.compare_methods:
                    compare_compression_methods(
                        q_unet,
                        cluster_quantizers,
                        sample_layers=5
                    )
                
                # Decompress weights for inference
                if args.use_compression and compressed_layers:
                    logger.info("\n🔄 Decompressing weights into model for inference...")
                    
                    model_compressor = ModelCompressor(cluster_quantizers)
                    model_compressor.compressed_layers = compressed_layers
                    
                    quantized_count = 0
                    for name in compressed_layers.keys():
                        weights = model_compressor.decompress_layer(name)
                        
                        for module_name, module in q_unet.named_modules():
                            if module_name == name and hasattr(module, 'org_weight'):
                                if hasattr(module, 'org_module'):
                                    module.org_module.weight.data = weights.cuda()
                                    quantized_count += 1
                                elif hasattr(module, 'weight'):
                                    module.weight.data = weights.cuda()
                                    quantized_count += 1
                                break
                    
                    logger.info(f"✅ Decompressed {quantized_count} layers for inference")
                
                else:
                    # No compression - use quantized weights directly
                    logger.info("\n📊 Using quantized weights (no compression)")
                    quantized_count = 0
                    
                    for name, quantizer in cluster_quantizers.items():
                        for module_name, module in q_unet.named_modules():
                            if module_name == name and hasattr(module, 'org_weight'):
                                org_weight = module.org_weight.cuda()
                                quantized_weight = quantizer.quantize_weights(org_weight)
                                
                                if hasattr(module, 'org_module'):
                                    module.org_module.weight.data = quantized_weight.clone()
                                    quantized_count += 1
                                elif hasattr(module, 'weight'):
                                    module.weight.data = quantized_weight.clone()
                                    quantized_count += 1
                                break
                    
                    logger.info(f"✅ Quantized {quantized_count} layers")
            
            # Calculate final statistics
            logger.info("\nCalculating final statistics...")
            if bitrate_info is not None:
                avg_bitwidth, avg_quant_value, analysis_results = average(q_unet, args, bitrate_info)
            else:
                avg_bitwidth, avg_quant_value, analysis_results = average(q_unet, args)
            
            logger.info(f"✅ Final average bitwidth: {avg_bitwidth:.3f} bits")
            logger.info(f"✅ Average quantized weight value: {avg_quant_value:.6f}")
            
            # Calculate actual model size
            if args.use_cluster_quant:
                total_bits, total_MB, comp_ratio = compute_actual_model_size(
                    q_unet,
                    cluster_quantizers=cluster_quantizers
                )
                
                logger.info("=" * 80)
                logger.info(f"📦 Actual model size: {total_MB:.2f} MB")
                logger.info(f"📉 True compression ratio (vs FP32): {comp_ratio:.2f}x")
                logger.info("=" * 80)

        elif args.use_cluster_quant:
            logger.info("\n" + "="*80)
            logger.info("APPLYING CLUSTER-BASED QUANTIZATION (NO ENTROPY MODEL)")
            logger.info("="*80)
            
            cluster_data = load_cluster_data(args.clustering_dir, q_unet)
            
            cluster_quantizers = {}
            quantized_count = 0
            
            for name, module in q_unet.named_modules():
                if not hasattr(module, 'org_weight'):
                    continue
                
                cluster_key = find_matching_layer(name, cluster_data['layer_clusters'])
                if cluster_key is None:
                    continue
                
                cluster_info = cluster_data['layer_clusters'][cluster_key]
                
                quantizer = ClusterBasedQuantizer(
                    clusters=cluster_info['clusters'],
                    densities=cluster_info['densities'],
                    assignments=cluster_info['assignments'],
                    num_clusters=len(cluster_info['clusters'])
                )
                cluster_quantizers[name] = quantizer
                
                org_weight = module.org_weight.cuda()
                quantized_weight = quantizer.quantize_weights(org_weight)
                
                if hasattr(module, 'org_module') and hasattr(module.org_module, 'weight'):
                    module.org_module.weight.data = quantized_weight.clone()
                    quantized_count += 1
                elif hasattr(module, 'weight'):
                    module.weight.data = quantized_weight.clone()
                    quantized_count += 1
            
            logger.info(f" Pre-quantized {quantized_count} layers")
            
            avg_bitwidth = calculate_average_bitwidth(q_unet, cluster_quantizers)
            logger.info(f"Average bitwidth: {avg_bitwidth:.3f} bits")
            
            # Calculate model size
            total_bits, total_MB, comp_ratio = compute_actual_model_size(
                q_unet,
                cluster_quantizers=cluster_quantizers
            )
            
            logger.info("=" * 80)
            logger.info(f"📦 Actual model size: {total_MB:.2f} MB")
            logger.info(f"📉 True compression ratio (vs FP32): {comp_ratio:.2f}x")
            logger.info("=" * 80)
        
        logger.info("\n🔧 Enabling quantization state...")
        q_unet.set_quant_state(weight_quant=False if args.use_cluster_quant else True, act_quant=True)
        setattr(model.model, 'diffusion_model', q_unet)

    logger.info("\n⚙️  Preparing model for sampling...")
    model.interval_seq = args.interval_seq
    model.model.reset_no_cache(no_cache=False)
    model.model.diffusion_model.model.time = 0
    imglogdir = "./error_dec/celeb/image1"
    os.makedirs(imglogdir, exist_ok=True)

    logger.info("\n🎨 Starting image generation...")
    logger.info(f"Generating {args.num_samples} samples...")
    logger.info(f"Batch size: {args.sample_batch}")
    logger.info(f"DDIM steps: {args.ddim_steps}")
    logger.info(f"Output directory: {imglogdir}")
    logger.info(f"Interval sequence: {model.interval_seq}")
    
    seed_everything(args.seed)
    run(model, imglogdir, eta=args.ddim_eta, n_samples=args.num_samples, 
        custom_steps=args.ddim_steps, batch_size=args.sample_batch)
    
    logger.info("\n✅ ALL DONE! Images saved to: " + imglogdir)