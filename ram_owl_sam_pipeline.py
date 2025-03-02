import argparse
import os
import numpy as np
import torch
import PIL.Image
from typing import List, Tuple, Dict, Optional

# RAM import
from nanoram.ram_trt_predictor import RAMTRTPredictor

# OWL import
from nanoowl.nanoowl.owl_predictor import OwlPredictor
from nanoowl.nanoowl.owl_drawing import draw_owl_output

# SAM import
from nanosam.nanosam.utils.predictor import Predictor as SamPredictor

class RAMOWLSAMPipeline:
    """
    End-to-end visual pipeline integrating RAM, OWL, and SAM:
    1. RAM generates image tags
    2. Uses tags to guide OWL to locate targets
    3. Uses OWL's boxes to guide SAM to segment masks
    """
    
    def __init__(
        self,
        # RAM model parameters
        ram_visual_encoder_engine: str,
        ram_tagging_head_engine: str,
        ram_tag_list_file: str,
        ram_tag_list_chinese_file: str,
        # OWL model parameters
        owl_model_dir: str,
        owl_image_encoder_engine: str,
        # SAM model parameters
        sam_image_encoder_engine: str,
        sam_mask_decoder_engine: str,
        # Optional parameters
        ram_class_threshold: float = 0.68,
        ram_image_size: int = 384,
        ram_max_batch_size: int = 1,
        owl_threshold: float = 0.1,
        device: str = "cuda"
    ):
        """Initialize the RAM-OWL-SAM pipeline
        
        Args:
            ram_visual_encoder_engine: Path to the RAM visual encoder TRT engine
            ram_tagging_head_engine: Path to the RAM tagging head TRT engine
            ram_tag_list_file: Path to the English tag list file (.txt)
            ram_tag_list_chinese_file: Path to the Chinese tag list file (.txt)
            ram_class_threshold: RAM tag classification threshold
            ram_image_size: RAM input image size
            ram_max_batch_size: RAM batch size
            
            owl_model_dir: OWL model directory
            owl_image_encoder_engine: Path to the OWL image encoder TRT engine
            owl_threshold: OWL detection threshold
            
            sam_image_encoder_engine: Path to the SAM image encoder TRT engine
            sam_mask_decoder_engine: Path to the SAM mask decoder TRT engine
            
            device: Running device ("cuda" or "cpu")
        """
        self.device = device
        
        # Initialize the RAM predictor
        self.ram_predictor = RAMTRTPredictor(
            visual_encoder_engine=ram_visual_encoder_engine,
            tagging_head_engine=ram_tagging_head_engine,
            tag_list_file=ram_tag_list_file,
            tag_list_chinese_file=ram_tag_list_chinese_file,
            class_threshold=ram_class_threshold,
            image_size=ram_image_size,
            device=device,
            max_batch_size=ram_max_batch_size
        )
        
        # Initialize the OWL predictor
        self.owl_predictor = OwlPredictor(
            owl_model_dir,
            image_encoder_engine=owl_image_encoder_engine
        )
        
        # Initialize the SAM predictor
        self.sam_predictor = SamPredictor(
            sam_image_encoder_engine,
            sam_mask_decoder_engine
        )
        
        # Set the OWL threshold
        self.owl_threshold = owl_threshold
    
    def process_image(
        self, 
        image_path: str,
        output_dir: str = "output",
        max_owl_labels: int = 5,
        save_intermediate: bool = True,
        selected_labels: Optional[List[str]] = None,
        language: str = "en"  # "en" 或 "zh"
    ) -> Dict:
        """Process the complete pipeline for a single image
        
        Args:
            image_path: Input image path
            output_dir: Output directory
            max_owl_labels: Maximum number of labels for OWL processing
            save_intermediate: Whether to save intermediate results
            selected_labels: Optional, specify the list of labels to process, if None uses the labels from RAM
            language: The language used, "en" for English, "zh" for Chinese
            
        Returns:
            A dictionary containing the processing results
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the image
        image = PIL.Image.open(image_path)
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        
        results = {
            "image_path": image_path,
            "ram_tags": {},
            "owl_results": [],
            "sam_masks": []
        }
        
        # Step 1: Use RAM to generate tags
        print(f"Step 1: Use RAM to generate image tags...")
        english_tags, chinese_tags = self.ram_predictor.generate_tag(image_path)
        
        # Select tags based on language
        tags = english_tags if language == "en" else chinese_tags
        results["ram_tags"]["english"] = english_tags
        results["ram_tags"]["chinese"] = chinese_tags
        
        # Split tags
        tag_list = tags.split(" | ")
        print(f"Detected tags: {tag_list}")
        
        # If specific labels are provided, use them
        if selected_labels:
            tag_list = selected_labels
            print(f"Using specified labels: {tag_list}")
        
        # Limit the number of tags
        tag_list = tag_list[:max_owl_labels]
        
        # Save RAM results
        if save_intermediate:
            with open(os.path.join(output_dir, f"{image_name}_ram_tags.txt"), "w", encoding="utf-8") as f:
                f.write(f"English tags: {english_tags}\n")
                f.write(f"中文标签: {chinese_tags}\n")
        
        # Step 2: Use RAM tags to guide OWL detection
        print(f"Step 2: Use {len(tag_list)} tags to guide OWL detection...")
        
        # Encode text
        text_encodings = self.owl_predictor.encode_text(tag_list)
        
        # Predict bounding boxes
        owl_output = self.owl_predictor.predict(
            image=image,
            text=tag_list,
            text_encodings=text_encodings,
            threshold=self.owl_threshold,
            pad_square=False
        )
        
        # Save OWL results
        if save_intermediate:
            owl_result_image = draw_owl_output(image.copy(), owl_output, text=tag_list, draw_text=True)
            owl_result_image.save(os.path.join(output_dir, f"{image_name}_owl_boxes.jpg"))
        
        # Build detection results
        for i, boxes in enumerate(owl_output["boxes"]):
            if len(boxes) > 0:
                label = tag_list[i]
                for box in boxes:
                    bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]  # [x0, y0, x1, y1]
                    score = float(box[4])
                    results["owl_results"].append({
                        "label": label,
                        "bbox": bbox,
                        "score": score
                    })
        
        # Step 3: Use OWL bounding boxes to guide SAM segmentation
        print(f"Step 3: Use {len(results['owl_results'])} detection boxes to guide SAM segmentation...")
        
        # Set the SAM image
        self.sam_predictor.set_image(image)
        
        # Create a segmentation for each detected object
        for i, detection in enumerate(results["owl_results"]):
            bbox = detection["bbox"]
            label = detection["label"]
            
            # Convert the bounding box to the point format expected by SAM
            points = np.array([
                [bbox[0], bbox[1]],  # Top-left corner
                [bbox[2], bbox[3]]   # Bottom-right corner
            ])
            point_labels = np.array([2, 3])  # 2=Top-left corner, 3=Bottom-right corner
            
            # Predict the mask
            mask, _, _ = self.sam_predictor.predict(points, point_labels)
            binary_mask = (mask[0, 0] > 0).detach().cpu().numpy()
            
            # Store results
            results["sam_masks"].append({
                "label": label,
                "bbox": bbox,
                "mask": binary_mask
            })
            
            # Save SAM results
            if save_intermediate:
                # Create a colored mask for visualization
                colored_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
                # Random color
                color = np.concatenate([np.random.randint(0, 255, 3), [128]])
                colored_mask[binary_mask] = color
                
                # Create a segmentation visualization
                mask_image = PIL.Image.fromarray(colored_mask, mode="RGBA")
                vis_image = image.copy().convert("RGBA")
                vis_image = PIL.Image.alpha_composite(vis_image, mask_image)
                
                # Draw the bounding box
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10))
                plt.imshow(np.array(vis_image))
                x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
                y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
                plt.plot(x, y, 'g-', linewidth=2)
                plt.title(f"{label}")
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f"{image_name}_sam_{i}_{label}.jpg"), 
                           bbox_inches='tight', pad_inches=0.0)
                plt.close()
        
        # Generate the final visualization image
        self._generate_final_visualization(
            image, results, os.path.join(output_dir, f"{image_name}_final.jpg"))
        
        print(f"Processing completed! Results saved to {output_dir}")
        return results
    
    def _generate_final_visualization(self, image, results, output_path):
        """Generate the final visualization image showing all results"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        plt.figure(figsize=(12, 12))
        plt.imshow(np.array(image))
        
        # Collect all masks
        masks = [result["mask"] for result in results["sam_masks"]]
        labels = [result["label"] for result in results["sam_masks"]]
        bboxes = [result["bbox"] for result in results["sam_masks"]]
        
        # Display masks
        for i, (mask, label, bbox) in enumerate(zip(masks, labels, bboxes)):
            # Random color
            color = np.random.rand(3)
            
            # Draw the mask
            masked_image = np.zeros_like(np.array(image), dtype=np.uint8)
            for c in range(3):
                masked_image[:, :, c] = np.where(mask, int(color[c] * 255), 0)
            
            plt.imshow(masked_image, alpha=0.5)
            
            # Draw the bounding box
            x0, y0, x1, y1 = bbox
            width, height = x1 - x0, y1 - y0
            plt.gca().add_patch(Rectangle((x0, y0), width, height, 
                                          fill=False, edgecolor=color, linewidth=2))
            
            # Add label
            plt.text(x0, y0-10, label, color=color, fontsize=12, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title("RAM-OWL-SAM Pipeline Results", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="RAM-OWL-SAM image processing pipeline")
    
    # Input/output parameters
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--selected_labels", type=str, default=None, 
                        help="Optional, specify the list of labels to process (comma-separated)")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"],
                        help="The language used, 'en' for English, 'zh' for Chinese")
    parser.add_argument("--max_owl_labels", type=int, default=5, 
                        help="Maximum number of labels for OWL processing")
    
    # RAM model parameters
    parser.add_argument("--ram_visual_encoder", type=str, required=True,
                        help="RAM visual encoder TRT engine path")
    parser.add_argument("--ram_tagging_head", type=str, required=True,
                        help="RAM tagging head TRT engine path")
    parser.add_argument("--ram_tag_list", type=str, required=True,
                        help="RAM English tag list file path")
    parser.add_argument("--ram_tag_list_chinese", type=str, required=True,
                        help="RAM Chinese tag list file path")
    parser.add_argument("--ram_threshold", type=float, default=0.68,
                        help="RAM tag classification threshold")
    parser.add_argument("--ram_image_size", type=int, default=384,
                        help="RAM input image size")
    
    # OWL model parameters
    parser.add_argument("--owl_model", type=str, required=True,
                        help="OWL model directory")
    parser.add_argument("--owl_image_encoder", type=str, required=True,
                        help="OWL image encoder TRT engine path")
    parser.add_argument("--owl_threshold", type=float, default=0.1,
                        help="OWL detection threshold")
    
    # SAM model parameters
    parser.add_argument("--sam_image_encoder", type=str, required=True,
                        help="SAM image encoder TRT engine path")
    parser.add_argument("--sam_mask_decoder", type=str, required=True,
                        help="SAM mask decoder TRT engine path")
    
    args = parser.parse_args()
    
    # Parse selected labels (if provided)
    selected_labels = None
    if args.selected_labels:
        selected_labels = [label.strip() for label in args.selected_labels.split(",")]
    
    # Create the pipeline
    pipeline = RAMOWLSAMPipeline(
        # RAM parameters
        ram_visual_encoder_engine=args.ram_visual_encoder,
        ram_tagging_head_engine=args.ram_tagging_head,
        ram_tag_list_file=args.ram_tag_list,
        ram_tag_list_chinese_file=args.ram_tag_list_chinese,
        ram_class_threshold=args.ram_threshold,
        ram_image_size=args.ram_image_size,
        
        # OWL parameters
        owl_model_dir=args.owl_model,
        owl_image_encoder_engine=args.owl_image_encoder,
        owl_threshold=args.owl_threshold,
        
        # SAM parameters
        sam_image_encoder_engine=args.sam_image_encoder,
        sam_mask_decoder_engine=args.sam_mask_decoder
    )
    
    # Process the image
    pipeline.process_image(
        image_path=args.image,
        output_dir=args.output_dir,
        max_owl_labels=args.max_owl_labels,
        selected_labels=selected_labels,
        language=args.language
    )

if __name__ == "__main__":
    main() 