"""
Clean CLIP checkpoint by removing double prefixes
"""

import torch
from pathlib import Path


def clean_clip_checkpoint(input_path: str, output_path: str):
    """
    Remove double prefixes from CLIP checkpoint
    
    Args:
        input_path: Original checkpoint
        output_path: Cleaned checkpoint
    """
    
    print("="*80)
    print("Cleaning CLIP Checkpoint")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("="*80 + "\n")
    
    # Load checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Original checkpoint has {len(state_dict)} keys\n")
    
    # Clean keys
    cleaned_state = {}
    
    for key, value in state_dict.items():
        # Remove double vision_model
        if 'vision_model.vision_model.' in key:
            new_key = key.replace('vision_model.vision_model.', 'vision_model.', 1)
            print(f"Cleaning: {key}")
            print(f"       → {new_key}")
            cleaned_state[new_key] = value
        
        # Remove double text_model
        elif 'text_model.text_model.' in key:
            new_key = key.replace('text_model.text_model.', 'text_model.', 1)
            print(f"Cleaning: {key}")
            print(f"       → {new_key}")
            cleaned_state[new_key] = value
        
        else:
            # Keep as is
            cleaned_state[key] = value
    
    print(f"\nCleaned checkpoint has {len(cleaned_state)} keys")
    
    # Save
    if isinstance(checkpoint, dict):
        checkpoint['model_state_dict'] = cleaned_state
        torch.save(checkpoint, output_path)
    else:
        torch.save(cleaned_state, output_path)
    
    print(f"\n✓ Saved cleaned checkpoint to: {output_path}")
    
    # Verify
    print("\n" + "="*80)
    print("Verification")
    print("="*80)
    
    verify_ckpt = torch.load(output_path, map_location='cpu')
    if 'model_state_dict' in verify_ckpt:
        verify_state = verify_ckpt['model_state_dict']
    else:
        verify_state = verify_ckpt
    
    print(f"✓ Loaded cleaned checkpoint")
    print(f"  Keys: {len(verify_state)}")
    
    # Check for double prefixes
    double_vision = [k for k in verify_state.keys() if 'vision_model.vision_model.' in k]
    double_text = [k for k in verify_state.keys() if 'text_model.text_model.' in k]
    
    if double_vision:
        print(f"  ⚠️  Still has {len(double_vision)} double vision_model prefixes")
    else:
        print(f"  ✓ No double vision_model prefixes")
    
    if double_text:
        print(f"  ⚠️  Still has {len(double_text)} double text_model prefixes")
    else:
        print(f"  ✓ No double text_model prefixes")
    
    # Sample keys
    print(f"\n  Sample keys:")
    for i, key in enumerate(list(verify_state.keys())[:5]):
        print(f"    {i+1}. {key}")
    
    print("\n" + "="*80)
    print("✓ Cleaning complete!")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        input_path = "checkpoints/clip_backbone/best_model.pt"
        output_path = "checkpoints/clip_backbone/best_model_cleaned.pth"
        print(f"Using defaults:")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}\n")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    
    clean_clip_checkpoint(input_path, output_path)