import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from typing import Dict, List, Optional
import torchvision.transforms as T
from datasets import load_dataset
from transformers import AutoTokenizer
import requests
from io import BytesIO

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
        # Local image directory
        image_dir: Optional[str] = None,
        caption_file: Optional[str] = None,
        
        # Huggingface dataset 
        dataset_name: Optional[str] = None,
        dataset_split: str = "train",
        image_column: str = "image",
        text_column: str = "text",
        streaming: bool = False,
        
        # Common Options 
        image_size: int = 224,
        augment: bool = False,
        tokenizer_name: str = "clicknext/phayathaibert",
        max_samples: Optional[int] = None
    ) :
        """_summary_

        Args:
            image_dir (str): Folder path containing images.
            caption_file (str): Path to the JSON file containing captions.
            image_size (int, optional): Size Image. Defaults to 224.
            augment (bool, optional): Whether to apply data augmentation. Defaults to False.
            tokenizer_name (str, optional): Name of the tokenizer to use. Defaults to "clicknext/phayathaibert".
        """
        
        # self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.max_samples = max_samples
        
        # load captions
        # with open(caption_file, 'r', encoding='utf-8') as f :
        #     self.captions_data = json.load(f)
            
        # self.image_filenames = list(self.captions_data.keys())
        
        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # setup image transformations
        self.transform = self._build_transforms()
        
        if dataset_name :
            self._load_hf_dataset(dataset_name, dataset_split, image_column, text_column, streaming)
        elif image_dir and caption_file :
            self._load_local_dataset(image_dir, caption_file)
        else :
            raise ValueError("Either dataset_name or both image_dir and caption_file must be provided.")

    def _load_hf_dataset(self, dataset_name: str, split: str, image_col: str, text_col: str, streaming: bool):
        print(f"Loading HF dataset: {dataset_name}, split: {split}")
        
        try:
            self.dataset = load_dataset(
                dataset_name, 
                split=split,
                streaming=streaming,
                cache_dir="data/huggingface"
            )
            
            # Check dataset structure
            print(f"Dataset features: {self.dataset.features if hasattr(self.dataset, 'features') else 'Unknown'}")
            
            # Test first item
            if not streaming:
                try:
                    # first_item = self.dataset[0]
                    # print(f"First item keys: {list(first_item.keys())}")
                    # print(f"Image column '{image_col}' exists: {image_col in first_item}")
                    # print(f"Text column '{text_col}' exists: {text_col in first_item}")
                    
                    # if image_col in first_item:
                    #     print(f"Image type: {type(first_item[image_col])}")
                    # if text_col in first_item:
                    #     print(f"Text type: {type(first_item[text_col])}")
                    #     if isinstance(first_item[text_col], list):
                    #         print(f"Text list length: {len(first_item[text_col])}")
                    #         if len(first_item[text_col]) > 0:
                    #             print(f"First text: {str(first_item[text_col][0])[:100]}")
                    pass
                                
                except Exception as e:
                    print(f"Error examining first item: {e}")
            
            if streaming:
                self.is_streaming = True
                self.data_items = self.dataset
            else:
                self.is_streaming = False
                self.data_items = self.dataset
                
                if self.max_samples:
                    print(f"Limiting to {self.max_samples} samples")
                    self.data_items = self.data_items.select(range(min(self.max_samples, len(self.data_items))))
            
            self.image_column = image_col
            self.text_column = text_col
            self.dataset_type = "hf"
            
            print(f"✓ Loaded HF dataset")
            try:
                dataset_size = len(self.data_items) if hasattr(self.data_items, '__len__') else "Unknown"
                print(f"  Samples: {dataset_size}")
            except:
                print(f"  Samples: Unknown (large dataset)")
            
        except Exception as e:
            print(f"Error loading HF dataset: {e}")
            raise
        
    def _load_local_dataset(self, image_dir: str, caption_file: str):
        """Load local dataset (original functionality)"""
        print(f"Loading local dataset from: {image_dir}")
        
        self.image_dir = image_dir
        
        # Load captions
        with open(caption_file, 'r', encoding='utf-8') as f:
            self.captions_data = json.load(f)
            
        self.image_filenames = list(self.captions_data.keys())
        
        # Apply max_samples limit
        if self.max_samples:
            self.image_filenames = self.image_filenames[:self.max_samples]
        
        self.dataset_type = "local"
        self.cap_per_image: int = 5
        
        print(f"✓ Loaded local dataset")
        print(f"  Samples: {len(self.image_filenames)}")
    
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
        if self.dataset_type == "hf" and hasattr(self, 'is_streaming') and self.is_streaming:
            # For streaming datasets, return a large number
            base_size = 1000000  # You might want to adjust this
        elif self.dataset_type == "hf":
            base_size = len(self.data_items)
        else:
            base_size = len(self.image_filenames)
        
        return base_size * self.cap_per_image
    
    def __getitem__(self, idx: int) -> Dict:
        original_idx = idx // self.cap_per_image
        caption_idx = idx % self.cap_per_image
        
        if self.dataset_type == "hf":
            return self._get_hf_item(idx)
        else:
            return self._get_local_item(idx)
        
    def _get_hf_item(self, idx: int) -> Dict:
        """Get item from Hugging Face dataset"""
        try:
            
            if self.max_samples and idx >= self.max_samples:
                idx = idx % self.max_samples
                
            if self.is_streaming:
                # For streaming, we need to iterate
                item = next(iter(self.data_items.skip(idx).take(1)))
            else:
                item = self.data_items[idx]
            
            # Debug: print item structure
            if idx == 0:  # Print structure for first item only
                print(f"Item keys: {list(item.keys())}")
                print(f"Image type: {type(item[self.image_column])}")
                print(f"Text type: {type(item[self.text_column])}")
                if isinstance(item[self.text_column], list):
                    print(f"Text length: {len(item[self.text_column])}")
                    print(f"First text: {item[self.text_column][0][:100] if item[self.text_column] else 'Empty'}")
            
            # Get image
            image = item[self.image_column]      
            
            # Handle image
            if image is None:
                print(f"Warning: None image at idx {idx}")
                image = Image.new('RGB', (self.image_size, self.image_size))
            elif hasattr(image, 'convert'):
                image = image.convert('RGB')
            elif isinstance(image, str):
                # Handle image path/URL
                if os.path.exists(image):
                    image = Image.open(image).convert('RGB')
                else:
                    print(f"Warning: Image path not found at idx {idx}: {image}")
                    image = Image.new('RGB', (self.image_size, self.image_size))
            else:
                print(f"Warning: Unknown image type at idx {idx}: {type(image)}")
                image = Image.new('RGB', (self.image_size, self.image_size))
            
            # Validate image size
            if image.size[0] == 0 or image.size[1] == 0:
                print(f"Warning: Invalid image size at idx {idx}: {image.size}")
                image = Image.new('RGB', (self.image_size, self.image_size))
            
            # Transform image
            image_tensor = self.transform(image)
            
            # Get text/caption
            caption_data = item[self.text_column]
            
            if caption_data is None:
                caption = "empty caption"
            elif isinstance(caption_data, list):
                if len(caption_data) > 0:
                    caption = str(caption_data[0]).strip()
                else:
                    caption = "empty caption"
            elif isinstance(caption_data, str):
                caption = caption_data.strip()
            else:
                caption = str(caption_data).strip()
            
            # Validate caption
            if not caption or caption == "":
                caption = "empty caption"
            
            return {
                'pixel_values': image_tensor,
                'caption': caption,
                'image_file': f"hf_item_{idx}"
            }
            
        except Exception as e:
            # แสดง error ที่แท้จริง
            print(f"Error processing HF item {idx}: {type(e).__name__}: {str(e)}")
            
            # ลองดู structure ของ item ถ้า error
            try:
                if 'item' in locals():
                    print(f"  Item keys: {list(item.keys()) if hasattr(item, 'keys') else 'No keys'}")
                    print(f"  Image column '{self.image_column}': {type(item.get(self.image_column, 'Missing'))}")
                    print(f"  Text column '{self.text_column}': {type(item.get(self.text_column, 'Missing'))}")
            except:
                pass
            
            # Return placeholder
            image = Image.new('RGB', (self.image_size, self.image_size))
            image_tensor = self.transform(image)
            
            return {
                'pixel_values': image_tensor,
                'caption': "placeholder caption",
                'image_file': f"error_item_{idx}"
            }
            
    def _get_local_item(self, idx: int) -> Dict:
        """Get item from local dataset (original functionality)"""
        image_file = self.image_filenames[idx]
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
        caption_data = self.captions_data[image_file]
        if isinstance(caption_data, dict):
            # Handle multiple captions or languages
            if 'captions' in caption_data:
                caption = caption_data['captions'][0]  # Take first caption
            elif 'th' in caption_data:
                caption = caption_data['th']
            elif 'en' in caption_data:
                caption = caption_data['en']
            else:
                caption = str(caption_data)
        else:
            caption = str(caption_data)
        
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
    

