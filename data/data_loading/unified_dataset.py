"""
Unified dataset for optical music recognition that loads Camera Primus data
and converts labels to BeKern format for compatibility with SMT models.
"""

import os
import logging
from typing import List, Union, Optional, Tuple
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np

from ..utils.format_converter import PrimusToBeKernConverter, load_primus_labels_from_file

logger = logging.getLogger(__name__)

class UnifiedDataset(data.Dataset):
    """
    A unified dataset class that loads Camera Primus images and labels,
    converts labels to BeKern format, and uses BeKern vocabulary.
    
    This allows using existing Camera Primus data with SMT (Sheet Music Transformer) models.
    """
    
    def __init__(self, 
                 data_paths: Union[str, List[str]], 
                 bekern_vocab_path: str,
                 mapping_file_path: Optional[str] = None,
                 transform: Optional[callable] = None):
        """
        Initialize the UnifiedDataset.
        
        Args:
            data_paths: Path(s) to directories containing Camera Primus data
            bekern_vocab_path: Path to BeKern vocabulary file (.npy format)
            mapping_file_path: Path to Primus-to-BeKern mapping JSON file
            transform: Optional image transforms to apply
        """
        # Handle both single path and multiple paths
        if isinstance(data_paths, str):
            self.data_paths = [data_paths]
        else:
            self.data_paths = data_paths
        
        self.transform = transform
        self.data = []  # List of (image_path, labels_path) tuples
        
        # Initialize format converter
        self.converter = PrimusToBeKernConverter(mapping_file_path)
        
        # Load BeKern vocabulary
        self.bekern_vocab_path = bekern_vocab_path
        self.vocabulary_to_index, self.index_to_vocabulary = self._load_bekern_vocabulary()
        
        # Load data samples
        self._load_data_samples()
        
        logger.info(f"Loaded {len(self.data)} samples from {len(self.data_paths)} directories")
        logger.info(f"BeKern vocabulary size: {len(self.vocabulary_to_index)}")
    
    def _load_bekern_vocabulary(self) -> Tuple[dict, dict]:
        """Load BeKern vocabulary from numpy file."""
        try:
            vocab_data = np.load(self.bekern_vocab_path, allow_pickle=True).item()
            
            # The SMT vocab files are word-to-index mappings
            if isinstance(vocab_data, dict):
                w2i = vocab_data
                i2w = {idx: word for word, idx in w2i.items()}
            else:
                raise ValueError("Vocabulary file should contain a dictionary")
                
            logger.info(f"Loaded BeKern vocabulary with {len(w2i)} tokens")
            return w2i, i2w
            
        except FileNotFoundError:
            logger.error(f"BeKern vocabulary file not found: {self.bekern_vocab_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading BeKern vocabulary: {e}")
            raise
    
    def _load_data_samples(self):
        """Load all data samples from the specified directories."""
        for data_path in self.data_paths:
            if not os.path.exists(data_path):
                logger.warning(f"Data path does not exist: {data_path}")
                continue
                
            # Iterate through each subdirectory (sample)
            for sample_dir in os.listdir(data_path):
                sample_dir_path = os.path.join(data_path, sample_dir)
                
                if not os.path.isdir(sample_dir_path):
                    continue
                
                image_file = None
                semantic_file = None
                
                # Find .png and .semantic files
                for file in os.listdir(sample_dir_path):
                    # Skip macOS metadata files
                    if file.startswith("._"):
                        continue
                    
                    if file.endswith(".png"):
                        image_file = os.path.join(sample_dir_path, file)
                    elif file.endswith(".semantic"):
                        semantic_file = os.path.join(sample_dir_path, file)
                
                # Check if we have both image and labels
                if image_file and semantic_file:
                    self.data.append((image_file, semantic_file))
                else:
                    missing = []
                    if not image_file:
                        missing.append("image")
                    if not semantic_file:
                        missing.append("labels")
                    logger.debug(f"Missing {', '.join(missing)} in {sample_dir_path}")
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (image_tensor, labels_tensor)
        """
        image_path, labels_path = self.data[index]
        
        # Load and process image
        image = self._load_and_process_image(image_path)
        
        # Load and convert labels
        labels = self._load_and_convert_labels(labels_path)
        
        return image, labels
    
    def _load_and_process_image(self, image_path: str) -> torch.Tensor:
        """Load and process an image."""
        try:
            # Read image and convert to grayscale
            image = Image.open(image_path).convert('L')
            
            # Resize to fixed height (128px) while preserving aspect ratio
            original_width, original_height = image.size
            aspect_ratio = original_width / original_height
            new_width = int(aspect_ratio * 128)
            
            # Apply resize transformation
            resize_transform = transforms.Resize((128, new_width))
            image = resize_transform(image)
            
            # Convert to tensor
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
            
            # Normalize (mean=0.5, std=0.5 for grayscale)
            normalize = transforms.Normalize((0.5,), (0.5,))
            image = normalize(image)
            
            # Apply additional transforms if specified
            if self.transform:
                image = self.transform(image)
                
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a dummy image tensor as fallback
            return torch.zeros(1, 128, 256)
    
    def _load_and_convert_labels(self, labels_path: str) -> torch.Tensor:
        """Load Camera Primus labels and convert to BeKern format."""
        try:
            # Load Primus tokens
            primus_tokens = load_primus_labels_from_file(labels_path)
            
            # Convert to BeKern format
            bekern_tokens = self.converter.convert_sequence_with_markers(primus_tokens)
            
            # Convert to indices using BeKern vocabulary
            indices = []
            for token in bekern_tokens:
                if token in self.vocabulary_to_index:
                    indices.append(self.vocabulary_to_index[token])
                else:
                    # Use padding token for unknown tokens
                    logger.debug(f"Unknown BeKern token: {token}")
                    if '<pad>' in self.vocabulary_to_index:
                        indices.append(self.vocabulary_to_index['<pad>'])
                    else:
                        indices.append(0)  # Fallback to index 0
            
            return torch.tensor(indices, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error loading labels {labels_path}: {e}")
            # Return a dummy label tensor as fallback
            return torch.tensor([0], dtype=torch.long)
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def get_conversion_stats(self) -> dict:
        """Get statistics about the token conversion process."""
        return self.converter.get_conversion_stats()
    
    def save_missing_tokens(self, output_file: str):
        """Save any missing tokens encountered during conversion."""
        self.converter.save_missing_tokens(output_file)
    
    def get_sample_info(self, index: int) -> dict:
        """Get information about a specific sample."""
        if index >= len(self.data):
            raise IndexError(f"Sample index {index} out of range")
        
        image_path, labels_path = self.data[index]
        
        # Load original labels for comparison
        primus_tokens = load_primus_labels_from_file(labels_path)
        bekern_tokens = self.converter.convert_sequence_with_markers(primus_tokens)
        
        return {
            "index": index,
            "image_path": image_path,
            "labels_path": labels_path,
            "original_tokens": primus_tokens,
            "converted_tokens": bekern_tokens,
            "num_original_tokens": len(primus_tokens),
            "num_converted_tokens": len(bekern_tokens)
        }

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for batch processing in UnifiedDataset.
    
    Pads images and labels to handle variable sizes in batches.
    """
    images, labels = zip(*batch)
    
    # Find maximum dimensions
    max_width = max([img.shape[2] for img in images])
    max_height = max([img.shape[1] for img in images])
    max_label_len = max([len(lbl) for lbl in labels])
    
    # Pad images
    padded_images = []
    for img in images:
        # Calculate padding
        padding_left = (max_width - img.shape[2]) // 2
        padding_right = max_width - img.shape[2] - padding_left
        padding_top = (max_height - img.shape[1]) // 2
        padding_bottom = max_height - img.shape[1] - padding_top
        
        # Apply padding
        padded = torch.nn.functional.pad(
            img, 
            (padding_left, padding_right, padding_top, padding_bottom), 
            "constant", 0
        )
        padded_images.append(padded)
    
    # Stack images
    padded_images = torch.stack(padded_images)
    
    # Pad labels
    padded_labels = []
    for lbl in labels:
        padding_len = max_label_len - len(lbl)
        padded_label = torch.cat([
            lbl, 
            torch.zeros(padding_len, dtype=torch.long)  # Pad with 0s
        ])
        padded_labels.append(padded_label)
    
    # Stack labels
    padded_labels = torch.stack(padded_labels)
    
    return padded_images, padded_labels
