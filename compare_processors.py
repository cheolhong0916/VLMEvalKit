"""
Compare image preprocessing results between Qwen2.5 and Qwen2 processors
"""

import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor
import argparse

def load_processors(model_path_25, model_path_2):
    """Load both Qwen2.5 and Qwen2 processors"""
    print("Loading processors...")
    
    try:
        processor_25 = AutoProcessor.from_pretrained(
            model_path_25,
            trust_remote_code=True
        )
        print(f"✅ Loaded Qwen2.5 processor: {type(processor_25.image_processor).__name__}")
    except Exception as e:
        print(f"❌ Failed to load Qwen2.5 processor: {e}")
        processor_25 = None
    
    try:
        processor_2 = AutoProcessor.from_pretrained(
            model_path_2,
            trust_remote_code=True
        )
        print(f"✅ Loaded Qwen2 processor: {type(processor_2.image_processor).__name__}")
    except Exception as e:
        print(f"❌ Failed to load Qwen2 processor: {e}")
        processor_2 = None
    
    return processor_25, processor_2


def compare_tensors(tensor1, tensor2, name="Tensor"):
    """Compare two tensors in detail"""
    print(f"\n{'='*60}")
    print(f"{name} Comparison")
    print(f"{'='*60}")
    
    if tensor1 is None or tensor2 is None:
        print("❌ One or both tensors are None")
        return False
    
    # Shape comparison
    print(f"Shape:")
    print(f"  Qwen2.5: {tensor1.shape}")
    print(f"  Qwen2  : {tensor2.shape}")
    
    if tensor1.shape != tensor2.shape:
        print("⚠️  Shapes are DIFFERENT!")
        return False
    
    # Dtype comparison
    print(f"\nData type:")
    print(f"  Qwen2.5: {tensor1.dtype}")
    print(f"  Qwen2  : {tensor2.dtype}")
    
    # Statistics (only for float tensors)
    if tensor1.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        print(f"\nStatistics (Qwen2.5):")
        print(f"  Min  : {tensor1.min().item():.6f}")
        print(f"  Max  : {tensor1.max().item():.6f}")
        print(f"  Mean : {tensor1.mean().item():.6f}")
        print(f"  Std  : {tensor1.std().item():.6f}")
        
        print(f"\nStatistics (Qwen2):")
        print(f"  Min  : {tensor2.min().item():.6f}")
        print(f"  Max  : {tensor2.max().item():.6f}")
        print(f"  Mean : {tensor2.mean().item():.6f}")
        print(f"  Std  : {tensor2.std().item():.6f}")
    else:
        print(f"\nStatistics (Qwen2.5):")
        print(f"  Min  : {tensor1.min().item()}")
        print(f"  Max  : {tensor1.max().item()}")
        
        print(f"\nStatistics (Qwen2):")
        print(f"  Min  : {tensor2.min().item()}")
        print(f"  Max  : {tensor2.max().item()}")
    
    # Difference
    diff = torch.abs(tensor1 - tensor2)
    print(f"\nAbsolute Difference:")
    print(f"  Min      : {diff.min().item():.6f}")
    print(f"  Max      : {diff.max().item():.6f}")
    print(f"  Mean     : {diff.mean().item():.6f}")
    print(f"  Std      : {diff.std().item():.6f}")
    
    # Are they identical?
    is_identical = torch.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-5)
    max_diff = diff.max().item()
    
    print(f"\n{'='*60}")
    if is_identical:
        print("✅ Tensors are IDENTICAL (within tolerance)")
    else:
        print(f"❌ Tensors are DIFFERENT (max diff: {max_diff:.6f})")
        
        # Show percentage of pixels that differ
        different_pixels = (diff > 1e-5).sum().item()
        total_pixels = diff.numel()
        percent_different = (different_pixels / total_pixels) * 100
        print(f"   {different_pixels}/{total_pixels} pixels differ ({percent_different:.2f}%)")
    print(f"{'='*60}\n")
    
    return is_identical


def process_and_compare(processor_25, processor_2, image_path, text_prompt):
    """Process image with both processors and compare"""
    
    print(f"\nProcessing image: {image_path}")
    print(f"Text prompt: {text_prompt}\n")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Create messages (Qwen format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt}
            ]
        }
    ]
    
    # Process with Qwen2.5
    print("\n--- Processing with Qwen2.5 ---")
    if processor_25:
        text_25 = processor_25.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs_25 = processor_25(
            text=[text_25],
            images=[image],
            return_tensors='pt'
        )
        print(f"Keys: {list(inputs_25.keys())}")
        print(f"Pixel values shape: {inputs_25['pixel_values'].shape}")
    else:
        inputs_25 = None
    
    # Process with Qwen2
    print("\n--- Processing with Qwen2 ---")
    if processor_2:
        text_2 = processor_2.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs_2 = processor_2(
            text=[text_2],
            images=[image],
            return_tensors='pt'
        )
        print(f"Keys: {list(inputs_2.keys())}")
        print(f"Pixel values shape: {inputs_2['pixel_values'].shape}")
    else:
        inputs_2 = None
    
    # Compare pixel values
    if inputs_25 and inputs_2:
        pixel_identical = compare_tensors(
            inputs_25['pixel_values'],
            inputs_2['pixel_values'],
            name="Pixel Values"
        )
        
        # Compare other keys if they exist
        common_keys = set(inputs_25.keys()) & set(inputs_2.keys())
        for key in common_keys:
            if key != 'pixel_values' and isinstance(inputs_25[key], torch.Tensor):
                compare_tensors(
                    inputs_25[key],
                    inputs_2[key],
                    name=key
                )
        
        return pixel_identical
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen2.5 and Qwen2 image processors")
    parser.add_argument("--model-path-25", type=str, required=True,
                        help="Path to Qwen2.5 model")
    parser.add_argument("--model-path-2", type=str, required=True,
                        help="Path to Qwen2 model")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to test image")
    parser.add_argument("--text", type=str, default="Describe this image.",
                        help="Text prompt to use")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Qwen2.5 vs Qwen2 Image Processor Comparison")
    print("="*80)
    
    # Load processors
    processor_25, processor_2 = load_processors(args.model_path_25, args.model_path_2)
    
    if not processor_25 or not processor_2:
        print("\n❌ Failed to load one or both processors. Exiting.")
        return
    
    # Compare
    is_identical = process_and_compare(
        processor_25, processor_2, args.image, args.text
    )
    
    print("\n" + "="*80)
    print("FINAL RESULT:")
    if is_identical:
        print("✅ Processors produce IDENTICAL results")
        print("   → No performance difference expected")
    else:
        print("❌ Processors produce DIFFERENT results")
        print("   → Performance difference LIKELY due to preprocessing mismatch")
        print("   → You MUST use the same processor for training and evaluation")
    print("="*80)


if __name__ == "__main__":
    main()