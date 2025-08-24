"""
Synthetic sheet music data generation using Verovio.
Adapted from SMT repository's VerovioGenerator.
"""

import re
import os
import cv2
import random
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageOps

import verovio
from cairosvg import svg2png
from wand.image import Image as IMG
import names
from wonderwords import RandomSentence


def clean_kern(krn: str, avoid_tokens: List[str] = None) -> str:
    """Clean kern notation by removing unwanted tokens."""
    if avoid_tokens is None:
        avoid_tokens = ['*Xped', '*staff1', '*staff2', '*tremolo', '*ped', '*Xtuplet', 
                       '*tuplet', "*Xtremolo", '*cue', '*Xcue', '*rscale:1/2', 
                       '*rscale:1', '*kcancel', '*below']
    
    krn = krn.split('\n')
    newkrn = []
    
    for idx, line in enumerate(krn):
        if not any([token in line.split('\t') for token in avoid_tokens]):
            if not all([token == '*' for token in line.split('\t')]):
                newkrn.append(line.replace("\n", ""))
                
    return "\n".join(newkrn)


def parse_kern(krn: str) -> str:
    """Parse kern notation into bekern format."""
    krn = clean_kern(krn)
    krn = re.sub(r'(?<=\=)\d+', '', krn)

    krn = krn.replace(" ", " <s> ")
    krn = krn.replace("\t", " <t> ")
    krn = krn.replace("\n", " <b> ")
    krn = krn.replace("·/", "")
    krn = krn.replace("·\\", "")

    krn = krn.strip().split(" ")[4:]
    
    return " ".join(krn)


def rfloat(start: float, end: float) -> float:
    """Generate random float between start and end."""
    return round(random.uniform(start, end), 2)


def rint(start: int, end: int) -> int:
    """Generate random integer between start and end."""
    return random.randint(start, end)


class SyntheticDataGenerator:
    """
    Generates synthetic sheet music images using Verovio from bekern data.
    """
    
    def __init__(self, bekern_data_path: str, output_format: str = 'bekern', seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.bekern_data_path = Path(bekern_data_path)
        self.output_format = output_format
        
        # Initialize Verovio
        verovio.enableLog(verovio.LOG_OFF)
        self.tk = verovio.toolkit()
        
        # Initialize text generators
        self.title_generator = RandomSentence()
        
        # Load bekern sequences for synthesis
        self.beat_db = self._load_bekern_sequences()
        
        # Setup texture paths (create default if not exists)
        self.textures_dir = Path("data/datasets/synthetic/textures")
        self._setup_textures()
    
    def _setup_textures(self):
        """Setup texture directory with default white texture if needed."""
        self.textures_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple white texture if no textures exist
        if not list(self.textures_dir.glob("*.png")):
            white_texture = Image.new('RGB', (2100, 2970), 'white')
            white_texture.save(self.textures_dir / "white_texture.png")
        
        self.textures = [str(f) for f in self.textures_dir.glob("*.png")]
    
    def _load_bekern_sequences(self) -> dict:
        """Load bekern sequences from your unified dataset."""
        from .format_converter import PrimusToBeKernConverter, load_primus_labels_from_file
        
        beats = {}
        converter = PrimusToBeKernConverter()
        
        # Look for Primus data in the specified path
        if not self.bekern_data_path.exists():
            print(f"Warning: Data path {self.bekern_data_path} does not exist")
            return beats
        
        # Iterate through all sample directories in the Primus dataset
        for sample_dir in self.bekern_data_path.iterdir():
            if not sample_dir.is_dir():
                continue
                
            # Find .semantic file
            semantic_files = list(sample_dir.glob("*.semantic"))
            if not semantic_files:
                continue
                
            semantic_file = semantic_files[0]
            
            try:
                # Load and convert Primus tokens to BeKern
                primus_tokens = load_primus_labels_from_file(str(semantic_file))
                bekern_tokens = converter.convert_sequence_with_markers(primus_tokens)
                
                # Convert to the format expected by the generator
                bekern_sequence = " ".join(bekern_tokens)
                
                # Check if this is a valid sequence (has time signature marker)
                if "*M" in bekern_sequence and bekern_sequence.count('*-') == 2:
                    # Extract time signature for organization
                    beat_marker_match = re.search(r'\*M\S*', bekern_sequence)
                    if beat_marker_match:
                        beat_marker = beat_marker_match.group()
                        beats.setdefault(beat_marker, []).append(bekern_sequence)
                        
            except Exception as e:
                print(f"Error processing {semantic_file}: {e}")
                continue
        
        # Remove time signatures with too few examples
        keys = list(beats.keys())
        for key in keys:
            if len(beats[key]) < 6:
                del beats[key]
        
        print(f"Loaded {sum(len(v) for v in beats.values())} bekern sequences across {len(beats)} time signatures")
        return beats
    
    def count_class_occurrences(self, svg_file: str, class_name: str) -> int:
        """Count occurrences of a CSS class in SVG."""
        root = ET.fromstring(svg_file)
        count = 0
        
        for element in root.iter():
            if class_name in element.get('class', ''):
                count += 1
        
        return count
    
    def find_image_cut(self, sample: np.ndarray) -> Optional[int]:
        """Find the bottom of the musical content for cropping."""
        height, _ = sample.shape[:2]
        
        for y in range(height - 1, -1, -1):
            if [0, 0, 0] in sample[y]:
                return y
        
        return None
    
    def render(self, music_sequence: str) -> str:
        """Render kern notation to SVG using Verovio."""
        self.tk.loadData(music_sequence)
        self.tk.setOptions({
            "pageWidth": 2100, 
            "footer": 'none',
            'barLineWidth': rfloat(0.3, 0.8), 
            'beamMaxSlope': rfloat(10, 20),
            'staffLineWidth': rfloat(0.1, 0.3), 
            'spacingStaff': rfloat(1, 12)
        })
        self.tk.getPageCount()
        svg = self.tk.renderToSVG()
        svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")
        return svg
    
    def convert_to_png(self, svg_file: str, cut: bool = False) -> np.ndarray:
        """Convert SVG to PNG image."""
        pngfile = svg2png(bytestring=svg_file, background_color='white')
        pngfile = cv2.imdecode(np.frombuffer(pngfile, np.uint8), -1)
        
        if cut:
            cut_height = self.find_image_cut(pngfile)
            if cut_height is not None:
                pngfile = pngfile[:cut_height + 10, :]
        
        return pngfile
    
    def inkify_image(self, sample: np.ndarray) -> Image.Image:
        """Apply ink-like effects to the image."""
        image = IMG.from_array(np.array(sample))
        paint = rfloat(0, 1)
        image.oil_paint(paint)
        return Image.fromarray(np.array(image))
    
    def generate_music_system_image(self, reduce_ratio: float = 0.5) -> Tuple[np.ndarray, List[str]]:
        """Generate a single system synthetic image."""
        if not self.beat_db:
            raise ValueError("No bekern sequences loaded. Please implement _load_bekern_sequences()")
        
        num_systems = 0
        
        while num_systems != 1:
            beat = random.choice(list(self.beat_db.keys()))
            music_seq = random.choice(self.beat_db[beat])
            
            # Convert bekern to kern format for Verovio
            render_sequence = "**kern\t**kern\n" + music_seq.replace(
                ' <b> ', '\n').replace(' <s> ', ' ').replace(' <t> ', '\t').replace('@', '').replace('·', '')
            
            image = self.render(render_sequence)
            num_systems = self.count_class_occurrences(svg_file=image, class_name='grpSym')
        
        x = self.convert_to_png(image, cut=True)
        x = cv2.cvtColor(np.array(x), cv2.COLOR_BGR2RGB)
        width = int(np.ceil(x.shape[1] * reduce_ratio))
        height = int(np.ceil(x.shape[0] * reduce_ratio))
        x = cv2.resize(x, (width, height))
        
        # Generate ground truth sequence based on format
        if self.output_format == "kern":
            gt_sequence = "".join(music_seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", "").replace('@', '').split(" ")
        elif self.output_format == "ekern":
            gt_sequence = "".join(music_seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace('@', '').split(" ")
        elif self.output_format == "bekern":
            gt_sequence = "".join(music_seq).replace("<s>", " <s> ").replace("<b>", " <b> ").replace("<t>", " <t> ").replace("·", " ").replace("@", " ").split(" ")
        
        return x, ['<bos>'] + [token for token in gt_sequence if token != ''] + ['<eos>']
    
    def generate_dataset(self, num_samples: int, output_dir: Path, system_level: bool = True):
        """Generate a complete synthetic dataset."""
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        for i in range(num_samples):
            try:
                if system_level:
                    image, gt_sequence = self.generate_music_system_image()
                else:
                    # TODO: Implement full page generation
                    raise NotImplementedError("Full page generation not yet implemented")
                
                # Save image
                image_filename = f"synthetic_{i:06d}.png"
                image_path = images_dir / image_filename
                cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # Save ground truth
                label_filename = f"synthetic_{i:06d}.json"
                label_path = labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    json.dump({
                        "transcription": " ".join(gt_sequence),
                        "format": self.output_format,
                        "image_path": str(image_path.relative_to(output_dir))
                    }, f, indent=2)
                
                metadata.append({
                    "id": i,
                    "image": image_filename,
                    "label": label_filename
                })
                
                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{num_samples} samples")
                    
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Generated {len(metadata)} samples successfully")