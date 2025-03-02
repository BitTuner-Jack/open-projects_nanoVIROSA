from dataclasses import dataclass
import torch
import numpy as np
from PIL import Image
import os
from typing import List, Tuple
from transform import get_transform
from torch2trt import TRTModule
import tensorrt as trt

@dataclass
class RAMVisualOutput:
    image_embeds: torch.Tensor
    label_embed: torch.Tensor

@dataclass
class RAMTaggingOutput:
    tag_indices: np.ndarray
    tag_scores: np.ndarray

class RAMTRTPredictor:
    def __init__(self,
                 visual_encoder_engine: str,
                 tagging_head_engine: str,
                 tag_list_file: str,
                 tag_list_chinese_file: str,
                 delete_tag_index: int = 4585,
                 class_threshold: float = 0.68,
                 image_size: int = 384,
                 device: str = "cuda",
                 max_batch_size: int = 1):
        """Initialize RAM TensorRT predictor
        
        Args:
            visual_encoder_engine: Path to visual encoder TRT engine
            tagging_head_engine: Path to tagging head TRT engine
            tag_list_file: Path to English tag list file (.txt)
            tag_list_chinese_file: Path to Chinese tag list file (.txt)
            delete_tag_index: Index of tag to be deleted (default: 4585)
            class_threshold: Threshold for tag classification (default: 0.68)
            image_size: Input image size (default: 384)
            device: Device to run on (default: "cuda")
            max_batch_size: Max batch size for TRT engines (default: 1)
        """
        self.device = torch.device(device)
        self.image_size = image_size
        self.max_batch_size = max_batch_size
        self.delete_tag_index = delete_tag_index
        self.class_threshold = class_threshold
        
        # Load tag lists
        def load_tag_list(file_path):
            """From. txt file to load a list of tags"""
            with open(file_path, 'r', encoding='utf-8') as f:
                return np.array([line.strip() for line in f.readlines()])
        
        self.tag_list = load_tag_list(tag_list_file)
        self.tag_list_chinese = load_tag_list(tag_list_chinese_file)
        self.num_class = len(self.tag_list)
        
        # Load TensorRT engines
        self.visual_encoder_engine = self.load_visual_encoder_engine(
            visual_encoder_engine, max_batch_size)
        self.tagging_head_engine = self.load_tagging_head_engine(
            tagging_head_engine, max_batch_size)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        transform = get_transform(image_size=self.image_size)
        image = transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        return image

    def encode_image(self, image: torch.Tensor) -> RAMVisualOutput:
        """Encode image using TensorRT engine"""
        # Ensure image is on the correct device
        image = image.to(self.device)
        with torch.no_grad():
            output = self.visual_encoder_engine([image])
            return RAMVisualOutput(
                image_embeds=output[0],
                label_embed=output[1]
            )

    def predict_tags(self, visual_output: RAMVisualOutput) -> RAMTaggingOutput:
        """Predict tags using TensorRT engine"""
        # Generate image attention mask
        image_atts = torch.ones(visual_output.image_embeds.shape[:-1], 
                                dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.tagging_head_engine([
                visual_output.label_embed,
                visual_output.image_embeds,
                image_atts
            ])
        
        return RAMTaggingOutput(
            tag_indices=output[0].cpu().numpy(),
            tag_scores=output[1].cpu().numpy()
        )

    def generate_tag(self, image_path: str) -> Tuple[str, str]:
        """Generate tags for an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (english_tags, chinese_tags)
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Encode image
        visual_output = self.encode_image(image)
        
        # Predict tags
        tag_output = self.predict_tags(visual_output)
        
        # Convert indices to tag strings
        tag = tag_output.tag_indices
        tag[:, self.delete_tag_index] = 0
        
        index = np.argwhere(tag[0] == 1)
        token = self.tag_list[index].squeeze(axis=1)
        token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
        
        tag_output = ' | '.join(token)
        tag_output_chinese = ' | '.join(token_chinese)
        
        return tag_output, tag_output_chinese

    def batch_generate_tags(self, image_paths: List[str]) -> List[Tuple[str, str]]:
        """Generate tags for multiple images in batch
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            List of tuples of (english_tags, chinese_tags)
        """
        results = []
        
        # Process images in batches of max_batch_size
        for i in range(0, len(image_paths), self.max_batch_size):
            batch_paths = image_paths[i:i+self.max_batch_size]
            batch_images = torch.cat([
                self.preprocess_image(path) for path in batch_paths
            ], dim=0)
            
            # Encode images
            visual_output = self.encode_image(batch_images)
            
            # Predict tags
            tag_output = self.predict_tags(visual_output)
            
            # Process each result in the batch
            tags = tag_output.tag_indices
            for b in range(tags.shape[0]):
                tag_b = tags[b:b+1]  # Keep batch dimension
                tag_b[:, self.delete_tag_index] = 0
                
                index = np.argwhere(tag_b[0] == 1)
                token = self.tag_list[index].squeeze(axis=1)
                token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
                
                tag_output_en = ' | '.join(token)
                tag_output_zh = ' | '.join(token_chinese)
                
                results.append((tag_output_en, tag_output_zh))
        
        return results

    @staticmethod
    def get_visual_encoder_input_names():
        return ["image"]

    @staticmethod 
    def get_visual_encoder_output_names():
        return ["image_embeds", "label_embed"]

    @staticmethod
    def get_tagging_head_input_names():
        return ["label_embed", "image_embeds", "attention_mask"]

    @staticmethod
    def get_tagging_head_output_names():
        return ["tag_indices", "tag_scores"]

    @staticmethod
    def load_visual_encoder_engine(engine_path: str, max_batch_size: int = 1):
        """Load TensorRT engine for visual encoder"""
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        base_module = TRTModule(
            engine,
            input_names=RAMTRTPredictor.get_visual_encoder_input_names(),
            output_names=RAMTRTPredictor.get_visual_encoder_output_names()
        )

        class Wrapper(torch.nn.Module):
            def __init__(self, base_module: TRTModule, max_batch_size: int):
                super().__init__()
                self.base_module = base_module
                self.max_batch_size = max_batch_size

            def forward(self, inputs):
                # Handle list input
                if isinstance(inputs, list):
                    image = inputs[0]  # Extract first element from list
                else:
                    image = inputs  # Use input directly
                    
                b = image.shape[0]
                results = []
                for start_idx in range(0, b, self.max_batch_size):
                    end_idx = min(b, start_idx + self.max_batch_size)
                    output = self.base_module(image[start_idx:end_idx])
                    results.append(output)
                return [
                    torch.cat([r[0] for r in results], dim=0),
                    torch.cat([r[1] for r in results], dim=0)
                ]

        return Wrapper(base_module, max_batch_size)

    @staticmethod
    def load_tagging_head_engine(engine_path: str, max_batch_size: int = 1):
        """Load TensorRT engine for tagging head"""
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        base_module = TRTModule(
            engine,
            input_names=RAMTRTPredictor.get_tagging_head_input_names(),
            output_names=RAMTRTPredictor.get_tagging_head_output_names()
        )

        class Wrapper(torch.nn.Module):
            def __init__(self, base_module: TRTModule, max_batch_size: int):
                super().__init__()
                self.base_module = base_module
                self.max_batch_size = max_batch_size

            def forward(self, inputs):
                # Handle list input
                if isinstance(inputs, list):
                    label_embed, image_embeds, attention_mask = inputs
                else:
                    # If not list, assume three arguments in order
                    label_embed, image_embeds, attention_mask = inputs
                    
                # Note: label_embed must keep its original shape [num_class,512]
                # Don't batch process label_embed
                
                b = image_embeds.shape[0]
                results = []
                for start_idx in range(0, b, self.max_batch_size):
                    end_idx = min(b, start_idx + self.max_batch_size)
                    # Important: label_embed doesn't need batch processing, pass in original shape
                    output = self.base_module(
                        label_embed,  # No slicing
                        image_embeds[start_idx:end_idx],
                        attention_mask[start_idx:end_idx]
                    )
                    results.append(output)
                
                # Return results
                return [
                    torch.cat([r[0] for r in results], dim=0),
                    torch.cat([r[1] for r in results], dim=0)
                ]

        return Wrapper(base_module, max_batch_size)

if __name__ == "__main__":
    # Initialize the predictor
    predictor = RAMTRTPredictor(
        visual_encoder_engine="ckpt/visual_encoder.engine",
        tagging_head_engine="ckpt/tagging_head.engine",
        tag_list_file="config/tag_list.txt",
        tag_list_chinese_file="config/tag_list_chinese.txt",
        image_size=384,
        max_batch_size=4
    )

    # Single image inference
    english_tags, chinese_tags = predictor.generate_tag("data/image.jpg")
    print(f"英文标签: {english_tags}")
    print(f"中文标签: {chinese_tags}")

    # # batch inference
    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    # results = predictor.batch_generate_tags(image_paths)
    # for i, (en_tags, zh_tags) in enumerate(results):
    #     print(f"图片 {i+1} - 英文: {en_tags}")
    #     print(f"图片 {i+1} - 中文: {zh_tags}")