import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import cv2
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd




script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "sam2"))
# Import works because sam2 is installed via pip
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- Configuration ---
IMAGE_SOURCE_DIR = os.path.join(script_dir, "train/balcony")
MASK_OUTPUT_DIR = os.path.join(script_dir, "annotated_masks")
DATA_OUTPUT_DIR = os.path.join(script_dir, "extracted_data")
BINARY_MASK_DIR = os.path.join(DATA_OUTPUT_DIR, "binary_masks")
ANNOTATIONS_DIR = os.path.join(DATA_OUTPUT_DIR, "annotations")
DATASET_SUMMARY_FILE = os.path.join(DATA_OUTPUT_DIR, "dataset_summary.csv")

# Create all necessary directories
for dir_path in [MASK_OUTPUT_DIR, DATA_OUTPUT_DIR, BINARY_MASK_DIR, ANNOTATIONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Real-world measurement scaling
PIXELS_PER_METER = 350.0  # Adjust based on your camera setup and typical distances

# Container size definitions (in square meters)
CONTAINER_SIZES = {
    'small': 0.09,      # 30cm x 30cm pot
    'medium': 0.25,     # 50cm x 50cm pot  
    'large': 0.64,      # 80cm x 80cm pot
    'extra_large': 1.0  # 1m x 1m raised bed
}

# Space type classification based on folder structure or manual input
SPACE_TYPES = ['rooftop', 'balcony', 'window', 'terrace', 'courtyard']

# Class map (added background color)
CLASS_MAP = {
    '1': {'name': 'growing_area', 'value': 1, 'color': [0, 255, 0]},  # Green
    '2': {'name': 'safety_zone', 'value': 2, 'color': [255, 0, 0]},   # Red
    '3': {'name': 'vertical_growing', 'value': 3, 'color': [0, 0, 255]}, # Blue
}
BACKGROUND_COLOR = [255, 255, 0]  # Bright Yellow for background

# --- Load the Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Loading SAM2 model...")

sam2_checkpoint = os.path.join(script_dir, "checkpoints", "sam2.1_hiera_large.pt")
model_cfg = os.path.join(script_dir, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")

if not os.path.exists(sam2_checkpoint):
    print(f"ERROR: Checkpoint file not found at: {sam2_checkpoint}")
    sys.exit(1)
if not os.path.exists(model_cfg):
    print(f"ERROR: Model config file not found at: {model_cfg}")
    sys.exit(1)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device)
mask_generator = SAM2AutomaticMaskGenerator(sam2)
print("Model loaded.")


class SpaceAnalyzer:
    """Analyzes space dimensions, shape, and capacity from masks"""
    
    @staticmethod
    def analyze_growing_area(binary_mask, pixels_per_meter=PIXELS_PER_METER):
        """Extract comprehensive measurements from binary mask"""
        if cv2.countNonZero(binary_mask) == 0:
            return None
            
        # Basic measurements
        total_pixels = cv2.countNonZero(binary_mask)
        total_area_sqm = total_pixels / (pixels_per_meter ** 2)
        
        # Find contours for shape analysis
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get largest contour (main growing area)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        width_m = w / pixels_per_meter
        height_m = h / pixels_per_meter
        
        # Shape analysis
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        
        # Circularity (4π×Area/Perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Aspect ratio
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        
        # Determine shape category
        if circularity > 0.8:
            shape_category = "circular"
        elif aspect_ratio < 1.5:
            shape_category = "square"
        elif aspect_ratio < 3.0:
            shape_category = "rectangular"
        else:
            shape_category = "long_narrow"
            
        # Calculate container capacity
        container_capacity = SpaceAnalyzer.calculate_container_capacity(total_area_sqm)
        
        # Get polygon coordinates for precise area
        polygon_coords = largest_contour.reshape(-1, 2).tolist()
        
        return {
            'total_area_sqm': round(total_area_sqm, 3),
            'total_pixels': int(total_pixels),
            'width_meters': round(width_m, 2),
            'height_meters': round(height_m, 2),
            'bounding_box': [int(x), int(y), int(w), int(h)],
            'shape_category': shape_category,
            'aspect_ratio': round(aspect_ratio, 2),
            'circularity': round(circularity, 3),
            'perimeter_pixels': int(perimeter),
            'polygon_coordinates': polygon_coords,
            'container_capacity': container_capacity,
            'contour_count': len(contours)
        }
    
    @staticmethod
    def calculate_container_capacity(total_area_sqm):
        """Calculate how many containers of each size can fit"""
        capacity = {}
        
        for size_name, size_area in CONTAINER_SIZES.items():
            # Use 70% of space to account for spacing between containers
            usable_area = total_area_sqm * 0.7
            capacity[size_name] = max(0, int(usable_area / size_area))
            
        return capacity
    
    @staticmethod
    def analyze_lighting_potential(image, growing_mask):
        """Analyze lighting conditions from image"""
        # Convert to HSV for better lighting analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract growing area pixels
        growing_pixels = hsv[growing_mask == 1]
        
        if len(growing_pixels) == 0:
            return None
            
        # Brightness analysis (V channel in HSV)
        brightness = growing_pixels[:, 2]
        avg_brightness = np.mean(brightness)
        
        # Lighting classification
        if avg_brightness > 180:
            lighting_category = "full_sun"
        elif avg_brightness > 120:
            lighting_category = "partial_sun"
        else:
            lighting_category = "shade"
            
        return {
            'average_brightness': int(avg_brightness),
            'lighting_category': lighting_category,
            'brightness_std': round(float(np.std(brightness)), 2)
        }


class EnhancedInteractiveAnnotator:
    def __init__(self, image_path, generated_masks):
        self.image_path = image_path
        self.image = np.array(Image.open(image_path).convert("RGB"))
        self.generated_masks = generated_masks
        self.semantic_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.fig, self.ax = plt.subplots(figsize=(15, 15))
        self.ax.set_title("Click to label masks. Press 's' to save & next, 'q' to quit.")
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.space_type = self.determine_space_type()
        self.update_display()

    def determine_space_type(self):
        """Determine space type from folder structure or user input"""
        # Try to determine from folder path
        path_lower = self.image_path.lower()
        for space_type in SPACE_TYPES:
            if space_type in path_lower:
                return space_type
                
        # If not found in path, ask user
        print(f"\nImage: {os.path.basename(self.image_path)}")
        print("Available space types:")
        for i, space_type in enumerate(SPACE_TYPES, 1):
            print(f"  {i}. {space_type}")
            
        while True:
            try:
                choice = input("Enter space type number (1-5): ")
                idx = int(choice) - 1
                if 0 <= idx < len(SPACE_TYPES):
                    return SPACE_TYPES[idx]
                else:
                    print("Invalid choice. Please enter 1-5.")
            except ValueError:
                print("Please enter a valid number.")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        try:
            x, y = int(event.xdata), int(event.ydata)
        except (TypeError, ValueError):
            return

        clicked_mask = None
        for m in self.generated_masks:
            if m['segmentation'][y, x]:
                if clicked_mask is None or m['area'] < clicked_mask['area']:
                    clicked_mask = m

        if clicked_mask is None:
            print("No mask found at this location.")
            return

        print("\n--- Mask Selected ---")
        for key, info in CLASS_MAP.items():
            print(f"  Enter '{key}' for {info['name']}")
        print("  Enter '0' to clear this region (set as background).")

        label_key = input("Enter label for the selected mask: ")

        if label_key in CLASS_MAP:
            class_info = CLASS_MAP[label_key]
            self.semantic_mask[clicked_mask['segmentation']] = class_info['value']
            print(f"Labeled region as '{class_info['name']}'.")
        elif label_key == '0':
            self.semantic_mask[clicked_mask['segmentation']] = 0
            print("Cleared region (set to background).")
        else:
            print("Invalid label. Ignoring.")

        self.update_display()

    def on_key_press(self, event):
        if event.key == 's':
            self.save_complete_annotation()
            plt.close(self.fig)
        elif event.key == 'q':
            plt.close(self.fig)
            self.semantic_mask = None

    def update_display(self):
        self.ax.clear()
        self.ax.imshow(self.image)

        # Overlay background color where mask is 0
        overlay = np.zeros((*self.semantic_mask.shape, 4), dtype=np.float32)
        background_mask = (self.semantic_mask == 0)
        color_with_alpha = np.array(BACKGROUND_COLOR + [100]) / 255.0
        overlay[background_mask] = color_with_alpha

        # Overlay labeled classes
        for key, info in CLASS_MAP.items():
            mask_for_class = (self.semantic_mask == info['value'])
            color_with_alpha = np.array(info['color'] + [150]) / 255.0
            overlay[mask_for_class] = color_with_alpha

        self.ax.imshow(overlay)
        self.ax.set_title("Click to label masks. Press 's' to save & next, 'q' to quit.")
        self.fig.canvas.draw()

    def save_complete_annotation(self):
        """Save all data: visual mask, binary mask, and extracted measurements"""
        filename = os.path.basename(self.image_path)
        name, _ = os.path.splitext(filename)
        
        # 1. Save visual overlay mask (your current approach)
        visual_mask_path = os.path.join(MASK_OUTPUT_DIR, f"{name}.png")
        Image.fromarray(self.semantic_mask).save(visual_mask_path)
        
        # 2. Save binary masks for each class
        binary_masks = {}
        for class_key, class_info in CLASS_MAP.items():
            class_mask = (self.semantic_mask == class_info['value']).astype(np.uint8) * 255
            binary_mask_path = os.path.join(BINARY_MASK_DIR, f"{name}_{class_info['name']}.png")
            Image.fromarray(class_mask).save(binary_mask_path)
            binary_masks[class_info['name']] = binary_mask_path
        
        # 3. Extract measurements and analysis
        growing_mask = (self.semantic_mask == 1).astype(np.uint8)
        space_analysis = SpaceAnalyzer.analyze_growing_area(growing_mask)
        lighting_analysis = SpaceAnalyzer.analyze_lighting_potential(self.image, growing_mask)
        
        # 4. Create comprehensive annotation data
        annotation_data = {
            'metadata': {
                'image_path': self.image_path,
                'image_name': filename,
                'annotation_date': datetime.now().isoformat(),
                'space_type': self.space_type,
                'image_dimensions': {
                    'width': self.image.shape[1],
                    'height': self.image.shape[0],
                    'channels': self.image.shape[2]
                }
            },
            'file_paths': {
                'original_image': self.image_path,
                'visual_mask': visual_mask_path,
                'binary_masks': binary_masks
            },
            'space_analysis': space_analysis,
            'lighting_analysis': lighting_analysis,
            'class_statistics': self._calculate_class_statistics()
        }
        
        # 5. Save annotation JSON
        annotation_path = os.path.join(ANNOTATIONS_DIR, f"{name}.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        print(f"\n✅ Complete annotation saved:")
        print(f"  📋 Visual mask: {visual_mask_path}")
        print(f"  🎯 Binary masks: {BINARY_MASK_DIR}/{name}_*.png")
        print(f"  📊 Analysis data: {annotation_path}")
        
        if space_analysis:
            print(f"\n📐 Space Analysis:")
            print(f"  Area: {space_analysis['total_area_sqm']:.2f} sq.m")
            print(f"  Shape: {space_analysis['shape_category']}")
            print(f"  Container capacity: {space_analysis['container_capacity']}")
        
        if lighting_analysis:
            print(f"  💡 Lighting: {lighting_analysis['lighting_category']}")

    def _calculate_class_statistics(self):
        """Calculate statistics for each labeled class"""
        stats = {}
        total_pixels = self.semantic_mask.size
        
        for class_key, class_info in CLASS_MAP.items():
            class_pixels = np.sum(self.semantic_mask == class_info['value'])
            percentage = (class_pixels / total_pixels) * 100
            
            stats[class_info['name']] = {
                'pixel_count': int(class_pixels),
                'percentage': round(percentage, 2)
            }
        
        # Background statistics
        background_pixels = np.sum(self.semantic_mask == 0)
        background_percentage = (background_pixels / total_pixels) * 100
        stats['background'] = {
            'pixel_count': int(background_pixels),
            'percentage': round(background_percentage, 2)
        }
        
        return stats


def update_dataset_summary():
    """Create/update a CSV summary of the entire dataset"""
    summary_data = []
    
    # Scan all annotation files
    for annotation_file in os.listdir(ANNOTATIONS_DIR):
        if not annotation_file.endswith('.json'):
            continue
            
        annotation_path = os.path.join(ANNOTATIONS_DIR, annotation_file)
        
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
            
            # Extract key information for CSV
            row = {
                'image_name': data['metadata']['image_name'],
                'space_type': data['metadata']['space_type'],
                'annotation_date': data['metadata']['annotation_date'],
                'total_area_sqm': data['space_analysis']['total_area_sqm'] if data['space_analysis'] else 0,
                'shape_category': data['space_analysis']['shape_category'] if data['space_analysis'] else 'none',
                'lighting_category': data['lighting_analysis']['lighting_category'] if data['lighting_analysis'] else 'unknown',
                'small_pots': data['space_analysis']['container_capacity']['small'] if data['space_analysis'] else 0,
                'medium_pots': data['space_analysis']['container_capacity']['medium'] if data['space_analysis'] else 0,
                'large_pots': data['space_analysis']['container_capacity']['large'] if data['space_analysis'] else 0,
                'growing_area_percentage': data['class_statistics']['growing_area']['percentage'],
            }
            summary_data.append(row)
            
        except Exception as e:
            print(f"Error processing {annotation_file}: {e}")
            continue
    
    # Save to CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(DATASET_SUMMARY_FILE, index=False)
        print(f"\n📊 Dataset summary updated: {DATASET_SUMMARY_FILE}")
        print(f"   Total annotated images: {len(summary_data)}")
        
        # Print quick statistics
        if len(summary_data) > 0:
            print(f"   Space types: {df['space_type'].value_counts().to_dict()}")
            print(f"   Average area: {df['total_area_sqm'].mean():.2f} sq.m")


def main():
    if not os.path.isdir(IMAGE_SOURCE_DIR):
        print(f"ERROR: Image source directory not found: {IMAGE_SOURCE_DIR}")
        sys.exit(1)

    image_paths = []
    for dirpath, _, filenames in os.walk(IMAGE_SOURCE_DIR):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(dirpath, filename))
    image_paths.sort()

    if not image_paths:
        print(f"No images found in {IMAGE_SOURCE_DIR} or its subfolders.")
        return

    print(f"Found {len(image_paths)} images to process.")

    for image_path in image_paths:
        name, _ = os.path.splitext(os.path.basename(image_path))
        annotation_file = os.path.join(ANNOTATIONS_DIR, f"{name}.json")
        
        if os.path.exists(annotation_file):
            print(f"⏭️  Skipping {os.path.basename(image_path)}, annotation already exists.")
            continue

        print(f"\n🔄 Processing {image_path}...")
        try:
            image_rgb = np.array(Image.open(image_path).convert("RGB"))
            generated_masks = mask_generator.generate(image_rgb)
            generated_masks = sorted(generated_masks, key=(lambda x: x['area']), reverse=False)
            
            annotator = EnhancedInteractiveAnnotator(image_path, generated_masks)
            plt.show()
            
            if annotator.semantic_mask is None:
                print("❌ Quitting annotation process.")
                break
                
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue
    
    # Update dataset summary
    update_dataset_summary()
    print("\n✅ Annotation session completed!")


if __name__ == '__main__':
    main()