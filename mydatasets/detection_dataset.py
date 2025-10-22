import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from typing import List, Dict, Tuple
import numpy as np
import torchvision.transforms as T
from transformers import AutoTokenizer

class CustomObjectDetectionDataset(Dataset) :
    """
    Format of annotations file :
    {
        "image_1.jpg" : [
            {
                "id": 1,
                "file_name": "image_1.jpg",
                "width": 800,
                "height": 600,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [x, y, width, height],
                "area": 40000,
                "iscrowd": 0
            }
        ],
        "categories" : [
            {"id": 0, "name": "category_1"},
            {"id": 1, "name": "category_2"}
        ]
    }
    """
    
    def __init__(
        self,
        annotations_file: str,
        images_dir: str,
        class_names: List[str],
        thai_class_names: List[str],
        image_size: int = 224,
        max_objects: int = 20,
        augment: bool = False,
        tokenizer_name: str = "clicknext/phayathaibert"
    ) :
        """
        Args:
            annotations_file: annotations file (COCO format)
            images_dir: directory containing images
            class_names: class names (English)
            thai_class_names: class names (Thai)
            image_size: image size
            max_objects: maximum number of objects per image
            augment: whether to use augmentation
            tokenizer_name: tokenizer name
        """
        self.images_dir = images_dir
        self.class_names = class_names
        self.thai_class_names = thai_class_names
        self.image_size = image_size
        self.max_objects = max_objects
        self.augment = augment
        
        # Load annotations
        print(f"Loading annotations from {annotations_file}...")
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
            
        # Build mappings
        self._build_mappings()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Setup transforms
        self.transform = self._build_transforms()
        
        print(f"✓ Loaded {len(self.images)} images with annotations")
        print(f"✓ Number of classes: {len(self.class_names)}")
        
    def _build_mappings(self):
        """ mappting for annotations """
        # image_id to annotations
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations'] :
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations :
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)
            
        # Filter image info with annotations
        self.images = [
            img_info for img_info in self.coco_data['images'] if img_info['id'] in self.image_id_to_annotations
        ]
        
    def _build_transforms(self) :
        """ build image transformations """
        transforms = []

        if self.augment :
            transforms.extend([
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                # T.RandomHorizontalFlip(p=0.5),
            ])
                
        transforms.extend([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return T.Compose(transforms)
    
    def __len__(self) :
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict :
        """
        Returns:
            {
                'pixel_values': Tensor [3, H, W],
                'boxes': Tensor [max_objects, 4],  # normalized [x_c, y_c, w, h]
                'labels': Tensor [max_objects],     # -1 for padding
                'text_queries': List[str],          # Thai class names
                'image_id': int,
                'num_objects': int
            }
        """
        # Get image info
        img_info = self.images[idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            orig_width, orig_height = image.size
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size))
            orig_width, orig_height = self.image_size, self.image_size
            
        # Get annotations
        annotations = self.image_id_to_annotations[img_info['id']]
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        for ann in annotations :
            # COCO bbox format : [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Convert to [x_c, y_c, w, h] normalized
            x_center = (x + w / 2) / orig_width
            y_center = (y + h / 2) / orig_height
            w_norm = w / orig_width
            h_norm = h / orig_height
            
            # Clip box values to [0, 1]
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            w_norm = np.clip(w_norm, 0, 1)
            h_norm = np.clip(h_norm, 0, 1)
            
            boxes.append([x_center, y_center, w_norm, h_norm])
            labels.append(ann['category_id'])
            
        # Limit to max_objects
        num_objects = len(boxes)
        if num_objects > self.max_objects :
            boxes = boxes[:self.max_objects]
            labels = labels[:self.max_objects]
            num_objects = self.max_objects
            
        # Create padding tensor 
        boxes_tensor = torch.zeros((self.max_objects, 4), dtype=torch.float32)
        labels_tensor = torch.full((self.max_objects,), -1, dtype=torch.long)
        
        if num_objects > 0:
            boxes_tensor[:num_objects] = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor[:num_objects] = torch.tensor(labels, dtype=torch.long)
            
        # Transform image
        image_tensor = self.transform(image)
        
        return {
            'pixel_values': image_tensor,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'text_queries': self.thai_class_names,
            'image_id': img_info['id'],
            'num_objects': num_objects
        }
        
def detection_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function สำหรับ detection dataset
    
    Returns:
        {
            'pixel_values': Tensor [B, 3, H, W],
            'boxes': Tensor [B, max_objects, 4],
            'labels': Tensor [B, max_objects],
            'input_ids': Tensor [B * num_classes, max_len],
            'attention_mask': Tensor [B * num_classes, max_len],
            'image_ids': List[int],
            'num_objects': Tensor [B]
        }
    """
    # Stack images
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Stack boxes and labels
    boxes = torch.stack([item['boxes'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Tokenize text queries
    text_queries = batch[0]['text_queries'] # same for all in batch
    tokenizer = AutoTokenizer.from_pretrained("clicknext/phayathaibert")
    text_inputs = tokenizer(
        text_queries,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors='pt'
    )
    
    return {
        'pixel_values': pixel_values,
        'boxes': boxes,
        'labels': labels,
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'image_ids': [item['image_id'] for item in batch],
        'num_objects': torch.tensor([item['num_objects'] for item in batch])
    }


def create_sample_coco_annotations(
    output_path: str,
    num_images: int = 10
):
    """
    สร้างไฟล์ตัวอย่าง COCO annotations
    
    Usage:
        create_sample_coco_annotations(
            'data/object_detection/annotations/train.json',
            num_images=10
        )
    """
    categories = [
        {"id": 0, "name": "chair"},
        {"id": 1, "name": "table"},
        {"id": 2, "name": "lamp"},
        {"id": 3, "name": "sofa"},
        {"id": 4, "name": "bed"},
    ]
    
    images = []
    annotations = []
    ann_id = 1
    
    for i in range(1, num_images + 1):
        # Create image entry
        images.append({
            "id": i,
            "file_name": f"room{i:03d}.jpg",
            "width": 640,
            "height": 480
        })
        
        # Create 2-5 random annotations per image
        num_objs = np.random.randint(2, 6)
        for _ in range(num_objs):
            # Random bbox
            x = np.random.randint(50, 400)
            y = np.random.randint(50, 300)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            
            annotations.append({
                "id": ann_id,
                "image_id": i,
                "category_id": np.random.randint(0, len(categories)),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✓ Created {len(images)} images with {len(annotations)} annotations")
    print(f"✓ Saved to {output_path}")

# =======================  Testing the Detection Dataset ======================= 


# print("Testing Detection Dataset...")

# # สร้าง sample annotations
# create_sample_coco_annotations(
#     'data/object_detection/annotations/train.json',
#     num_images=5
# )

# dataset = CustomObjectDetectionDataset(
#     annotations_file='data/object_detection/annotations/train.json',
#     images_dir='data/object_detection/images/train',
#     class_names=[
#         "door", "cabinetDoor", "refrigeratorDoor", "window", "chair",
#         "table", "cabinet", "couch", "openedDoor", "pole"
#       ],
#     thai_class_names=[
#         "ประตู","ประตูตู้", "ประตูตู้เย็น", "หน้าต่าง", "เก้าอี้", 
#         "โต๊ะ", "ตู้", "โซฟา", "ประตูเปิด", "เสา"
#       ],
#     image_size=224,
#     augment=True
# )   

# print(f"Dataset size: {len(dataset)}")

# if len(dataset) > 0:
#     sample = dataset[0]
#     print(f"\nSample keys: {sample.keys()}")
#     print(f"Image shape: {sample['pixel_values'].shape}")
#     print(f"Boxes shape: {sample['boxes'].shape}")
#     print(f"Labels shape: {sample['labels'].shape}")
#     print(f"Num objects: {sample['num_objects']}")
#     print(f"Thai queries: {sample['text_queries']}")