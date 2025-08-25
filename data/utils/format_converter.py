"""
Format converter module for converting Camera Primus tokens to BeKern format.
Uses a JSON mapping file for simple dictionary-based conversion.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class PrimusToBeKernConverter:
    """Converts Camera Primus format tokens to BeKern format using JSON mappings."""
    
    def __init__(self, mapping_file_path: Optional[str] = None):
        """
        Initialize the converter with the mapping file.
        
        Args:
            mapping_file_path: Path to the JSON mapping file. If None, uses default.
        """
        if mapping_file_path is None:
            # Default path relative to this file
            current_dir = Path(__file__).parent
            mapping_file_path = current_dir / "primus_to_bekern_mapping.json"
        
        self.mapping_file_path = Path(mapping_file_path)
        self.token_mapping = self._load_mapping()
        self.missing_tokens = set()
        
    def _load_mapping(self) -> Dict[str, str]:
        """Load the token mapping from JSON file."""
        try:
            with open(self.mapping_file_path, 'r') as f:
                mapping = json.load(f)
            
            # Filter out empty mappings
            filtered_mapping = {k: v for k, v in mapping.items() if v}
            
            logger.info(f"Loaded {len(filtered_mapping)} token mappings from {self.mapping_file_path}")
            return filtered_mapping
            
        except FileNotFoundError:
            logger.error(f"Mapping file not found: {self.mapping_file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in mapping file: {e}")
            return {}
    
    def convert_token(self, primus_token: str) -> Optional[str]:
        """
        Convert a single Primus token to BeKern format.
        
        Args:
            primus_token: The Camera Primus token to convert
            
        Returns:
            BeKern token if mapping exists, None otherwise
        """
        bekern_token = self.token_mapping.get(primus_token)
        
        if bekern_token is None:
            self.missing_tokens.add(primus_token)
            logger.debug(f"No mapping found for token: {primus_token}")
            
        return bekern_token
    
    def convert_sequence(self, primus_tokens: List[str]) -> List[str]:
        """
        Convert a sequence of Primus tokens to BeKern format.
        
        Args:
            primus_tokens: List of Camera Primus tokens
            
        Returns:
            List of BeKern tokens (unmappable tokens are skipped)
        """
        bekern_tokens = []
        
        for token in primus_tokens:
            bekern_token = self.convert_token(token)
            if bekern_token is not None:
                bekern_tokens.append(bekern_token)
        
        return bekern_tokens
    
    def convert_sequence_with_markers(self, primus_tokens: List[str]) -> List[str]:
        """
        Convert a sequence and add BeKern sequence markers (<bos>, <eos>).
        
        Args:
            primus_tokens: List of Camera Primus tokens
            
        Returns:
            List of BeKern tokens with sequence markers
        """
        bekern_tokens = self.convert_sequence(primus_tokens)
        
        # Add sequence markers as expected by SMT models
        return ["<bos>"] + bekern_tokens + ["<eos>"]
    
    def get_conversion_stats(self) -> Dict[str, int]:
        """
        Get statistics about the conversion process.
        
        Returns:
            Dictionary with conversion statistics
        """
        return {
            "total_mappings": len(self.token_mapping),
            "missing_tokens_encountered": len(self.missing_tokens),
            "missing_tokens": list(self.missing_tokens)
        }
    
    def save_missing_tokens(self, output_file: str):
        """
        Save encountered missing tokens to a file for manual mapping.
        
        Args:
            output_file: Path to save missing tokens
        """
        if not self.missing_tokens:
            logger.info("No missing tokens to save")
            return
            
        with open(output_file, 'w') as f:
            json.dump({
                "missing_tokens": list(self.missing_tokens),
                "count": len(self.missing_tokens),
                "note": "These tokens were encountered but have no mapping in the conversion file"
            }, f, indent=2)
        
        logger.info(f"Saved {len(self.missing_tokens)} missing tokens to {output_file}")

def load_primus_labels_from_file(file_path: str) -> List[str]:
    """
    Load Camera Primus labels from a .semantic file.
    
    Args:
        file_path: Path to the .semantic file
        
    Returns:
        List of token strings
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # Split by tabs and filter out empty strings
            tokens = [token for token in content.split('\t') if token]
            return tokens
    except FileNotFoundError:
        logger.error(f"Label file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading label file {file_path}: {e}")
        return []


def load_bekern_labels_from_file(file_path: str) -> List[str]:
    """
    Load BeKern labels from a .semantic file.
    
    BeKern files have a different format - they start with **ekern_1.0 header
    and contain tab-separated data organized in columns (staves).
    
    Args:
        file_path: Path to the .semantic file
        
    Returns:
        List of token strings
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        tokens = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip header lines that start with **
            if line.startswith('**'):
                continue
                
            # Split by tabs and add all non-empty tokens
            line_tokens = [token for token in line.split('\t') if token.strip()]
            tokens.extend(line_tokens)
            
        return tokens
        
    except FileNotFoundError:
        logger.error(f"BeKern label file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading BeKern label file {file_path}: {e}")
        return []

def example_usage():
    """Example of how to use the converter."""
    # Create converter
    converter = PrimusToBeKernConverter()
    
    # Example conversion
    primus_tokens = [
        "clef-G2",
        "keySignature-CM", 
        "timeSignature-4/4",
        "note-C4_quarter",
        "note-D4_quarter", 
        "note-E4_half",
        "barline"
    ]
    
    print("Original Primus tokens:")
    print(primus_tokens)
    
    bekern_tokens = converter.convert_sequence_with_markers(primus_tokens)
    print("\nConverted BeKern tokens:")
    print(bekern_tokens)
    
    print("\nConversion stats:")
    stats = converter.get_conversion_stats()
    print(f"Total mappings available: {stats['total_mappings']}")
    print(f"Missing tokens encountered: {stats['missing_tokens_encountered']}")
    
    return bekern_tokens

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    example_usage()