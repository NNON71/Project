import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from typing import Dict, List
import torchvision.transforms as T
from transformers import AutoTokenizer

class CLIPPretrainingDataset(Dataset) :
    """
    Format of captions file :
    {
        "image_1.jpg": {
            "captions": [
                "A caption describing image 1.",
                "Another caption for image 1."
            ]
        },
        "image_2.jpg": {
            "captions": [
                "A caption describing image 2.",
                "Another caption for image 2."
            ]
        }
    }
    """
    
    def __init__(
        self,
        image_dir: str,
        caption_file: str,
        image_size: int = 224,
        augment: bool = False,
        tokenizer_name: str = "clicknext/phayathaibert"
    ) :
        """_summary_

        Args:
            image_dir (str): Folder path containing images.
            caption_file (str): Path to the JSON file containing captions.
            image_size (int, optional): Size Image. Defaults to 224.
            augment (bool, optional): Whether to apply data augmentation. Defaults to False.
            tokenizer_name (str, optional): Name of the tokenizer to use. Defaults to "clicknext/phayathaibert".
        """
        
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        
        # load captions
        with open(caption_file, 'r', encoding='utf-8') as f :
            self.captions_data = json.load(f)
            
        self.image_filenames = list(self.captions_data.keys())
        
        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # setup image transformations
        self.transform = self._build_transforms()
        
    def _build_transforms(self) :
        transforms = []
        
        if self.augment:
            # Training augmentation
            transforms.extend([
                T.RandomResizedCrop(
                    self.image_size, 
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
            ])
        else:
            # Validation/test
            transforms.append(
                T.Resize((self.image_size, self.image_size))
            )
            
        # Common transforms
        transforms.extend([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return T.Compose(transforms)
    
    def __len__(self) :
        return len(self.image_filenames)
    
    def __getitem__(self, idx: int) -> Dict :
        """
        Return:
         {
             'pixel_values': Tensor [3, H, W],
             'caption': str,
             'image_file': str
         }
        """ 
        
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return placeholder
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        # Transform
        image_tensor = self.transform(image)
        
        # Get caption
        caption_data = self.captions[image_file]
        if self.use_thai and 'th' in caption_data:
            caption = caption_data['th']
        else:
            caption = caption_data.get('en', '')
        
        return {
            'pixel_values': image_tensor,
            'caption': caption,
            'image_file': image_file
        }
        
def clip_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function สำหรับ CLIP dataset
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        {
            'pixel_values': Tensor [B, 3, H, W],
            'input_ids': Tensor [B, max_len],
            'attention_mask': Tensor [B, max_len],
            'captions': List[str]
        }
    """
    # Stack images
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Get captions
    captions = [item['caption'] for item in batch]
    
    # Tokenize captions
    tokenizer = AutoTokenizer.from_pretrained("clicknext/phayathaibert")
    text_inputs = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=77,  # CLIP standard
        return_tensors='pt'
    )
    
    return {
        'pixel_values': pixel_values,
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'captions': captions
    }