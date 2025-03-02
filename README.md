# Vision Nano RAM-OWL-SAM visual pipeline 🚀

![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.1-yellowgreen)
![TensorRT](https://img.shields.io/badge/TensorRT-10.7-orange)

**Vi**sion Nano **R**AM-**O**WL-**SA**M is an end-to-end visual AI pipeline that integrates three powerful AI visual models to complete the entire process from image recognition, object detection to instance segmentation. The pipeline uses TensorRT to accelerate inference, and can efficiently process images in various complex scenarios.

## 📦 Pipeline Components

| Nano model | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| **RAM**🏷️   | High-performance image tag generation model, capable of identifying objects, scenes, and attributes in images |
| **OWL**🔍   | Open-vocabulary object localization model, capable of locating objects in images based on text descriptions |
| **SAM**✂️   | Powerful image segmentation model, capable of generating high-quality object masks |

## 🛠️ Installation Requirements

### ⚙️ Dependencies

```bash
# Basic dependencies
pip install torch torchvision numpy Pillow matplotlib 

# TensorRT related dependencies
pip install tensorrt torch2trt
```

### 🧠 Pre-trained models

You need to prepare the following TensorRT engine files:

- **RAM model:**
  - Visual encoder engine (`ram_visual_encoder.engine`)
  - Tagging head engine (`ram_tagging_head.engine`)
  - English tag list file (`ram_tag_list.txt`)
  - Chinese tag list file (`ram_tag_list_chinese.txt`)

- **OWL model:**
  - Model directory (contains configuration files)
  - Image encoder engine (`owl_image_encoder.engine`)

- **SAM模型:**
  - Image encoder engine (`sam_image_encoder.engine`)
  - Mask decoder engine (`sam_mask_decoder.engine`)

## 🚀 Usage

### 🎯 Basic usage

```bash
python ram_owl_sam_pipeline.py \
  --image image.jpg \
  --output_dir output \
  --ram_visual_encoder ram_visual_encoder.engine \
  --ram_tagging_head ram_tagging_head.engine \
  --ram_tag_list ram_tag_list.txt \
  --ram_tag_list_chinese ram_tag_list_chinese.txt \
  --owl_model owl_model \
  --owl_image_encoder owl_image_encoder.engine \
  --sam_image_encoder sam_image_encoder.engine \
  --sam_mask_decoder sam_mask_decoder.engine
```

### ⚡ Advanced options

```bash
python ram_owl_sam_pipeline.py \
  --image image.jpg \
  --output_dir output \
  --ram_visual_encoder ram_visual_encoder.engine \
  --ram_tagging_head ram_tagging_head.engine \
  --ram_tag_list ram_tag_list.txt \
  --ram_tag_list_chinese ram_tag_list_chinese.txt \
  --owl_model owl_model \
  --owl_image_encoder owl_image_encoder.engine \
  --sam_image_encoder sam_image_encoder.engine \
  --sam_mask_decoder sam_mask_decoder.engine \
  --language zh \
  --selected_labels "猫,狗,人" \
  --max_owl_labels 10 \
  --ram_threshold 0.5 \
  --owl_threshold 0.2
```

## 📊 Parameter description

### 📥 Input/output parameters

- `--image`: Input image path
- `--output_dir`: Output directory, default is "output"
- `--selected_labels`: Optional, specify the list of labels to process (comma-separated)
- `--language`: The language used, 'en' for English, 'zh' for Chinese, default is 'en'
- `--max_owl_labels`: The maximum number of labels for OWL processing, default is 5

### 🏷️ RAM model parameters

- `--ram_visual_encoder`: RAM visual encoder TRT engine path
- `--ram_tagging_head`: RAM tagging head TRT engine path
- `--ram_tag_list`: RAM English tag list file path
- `--ram_tag_list_chinese`: RAM Chinese tag list file path
- `--ram_threshold`: RAM tag classification threshold, default is 0.68
- `--ram_image_size`: RAM input image size, default is 384

### 🔍 OWL model parameters

- `--owl_model`: OWL model directory
- `--owl_image_encoder`: OWL image encoder TRT engine path
- `--owl_threshold`: OWL detection threshold, default is 0.1

### ✂️ SAM model parameters

- `--sam_image_encoder`: SAM image encoder TRT engine path
- `--sam_mask_decoder`: SAM mask decoder TRT engine path

## 📁 Output results

The pipeline will generate the following files after processing a single image:

1. `{image_name}_ram_tags.txt` - RAM generated English and Chinese tags
2. `{image_name}_owl_boxes.jpg` - OWL detection bounding box visualization
3. `{image_name}_sam_{index}_{label}.jpg` - SAM segmentation mask for each detected object
4. `{image_name}_final.jpg` - Final integrated visualization result, containing all detected object bounding boxes and segmentation masks

## ⚙️ Performance and configuration

The pipeline uses TensorRT optimized models for inference, which can run efficiently on NVIDIA GPUs supporting CUDA. You can adjust the performance by:

- Adjusting the `max_batch_size` parameter of RAM to batch process multiple images
- Adjusting the `ram_threshold` and `owl_threshold` parameters to balance recall and precision
- Limiting the number of processed labels through the `max_owl_labels` parameter to improve speed

## ❓ Common questions and answers

### How to improve detection accuracy?

- Adjusting the `ram_threshold` parameter of RAM can control the quality of generated tags
- Adjusting the `owl_threshold` parameter of OWL can control the accuracy of detected bounding boxes
- If a specific object is not detected, you can directly specify the label to be detected using the `selected_labels` parameter

### What if the processing speed is too slow?

- Reduce the value of the `max_owl_labels` parameter to limit the number of processed labels
- Ensure that the correct TensorRT engine is used
- Use a higher performance GPU

### What if the memory is insufficient?

- Reduce the number of processed labels
- Lower the image resolution
- Use a GPU with more memory

## 🖼️ Application scenarios

This pipeline is suitable for various visual AI application scenarios:

1. **Image content analysis**: Automatically identify and segment the main objects in the image
2. **Visual content creation**: Provide automatic object segmentation functionality for creative tools
3. **Photo editing**: Automatically identify and separate the main subject from a photo
4. **Data annotation**: Accelerate the creation of computer vision datasets
5. **Visual search**: Smart search and classification based on image content
6. **Robot vision**: Provide scene understanding capabilities for robots

## 🧩 Code example

Here is an example of using the Python API to process an image:

```python
from ram_owl_sam_pipeline import RAMOWLSAMPipeline

# Initialize the pipeline
pipeline = RAMOWLSAMPipeline(
    # RAM parameters
    ram_visual_encoder_engine="ckpt/ram_visual_encoder.engine",
    ram_tagging_head_engine="ckpt/ram_tagging_head.engine",
    ram_tag_list_file="nanoram/config/ram_tag_list.txt",
    ram_tag_list_chinese_file="nanoram/config/ram_tag_list_chinese.txt",
    
    # OWL parameters
    owl_model_dir="ckpt/owl_model",
    owl_image_encoder_engine="ckpt/owl_image_encoder.engine",
    
    # SAM parameters
    sam_image_encoder_engine="ckpt/sam_image_encoder.engine",
    sam_mask_decoder_engine="ckpt/sam_mask_decoder.engine"
)

# Process a single image
results = pipeline.process_image(
    image_path="image.jpg",
    output_dir="output",
    language="zh",  # Use Chinese tags
    selected_labels=["猫", "狗"]  # Process specific labels
)

# View results
print(f"Detected tags: {results['ram_tags']}")
print(f"Number of detected objects: {len(results['owl_results'])}")
print(f"Number of generated masks: {len(results['sam_masks'])}")
```

## 🔄 Batch processing example

Here is an example of batch processing multiple images:

```python
import os
from ram_owl_sam_pipeline import RAMOWLSAMPipeline

# Initialize the pipeline
pipeline = RAMOWLSAMPipeline(
    # Configure parameters as before
    ...
)

# Batch process images
image_dir = "path/to/images"
output_dir = "output"

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(f"Processing image: {image_file}")
    
    # Call the processing function
    pipeline.process_image(
        image_path=image_path,
        output_dir=output_dir,
        language="zh"
    )

print(f"Batch processing completed, processed {len(image_files)} images.")
```

## 📝 Notes

- All model engine files should be optimized by TensorRT and converted to `.engine` format
- Ensure sufficient GPU memory to run all three models
- The pipeline assumes that the input sizes of different models have been appropriately configured
- The RAM model requires a bilingual label table to support multi-language output


## 🚧 Future plans

- Implement video processing functionality
- Integrate tracking functionality to process video sequences
- Optimize model inference performance

## 📚 Citation and Acknowledgements

This project is based on the following open-source models:

- RAM (Recognize Anything Model): [https://github.com/xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything)
- Nanoowl(Open-Vocabulary Object Localization): [https://github.com/NVIDIA/nanoowl](https://github.com/NVIDIA/nanoowl)
- Nanosam(Segment Anything Model): [https://github.com/NVIDIA/nanosam](https://github.com/NVIDIA/nanosam)

Thanks for the contributions of these excellent projects.

## 📜 License

This project is licensed under the MIT License. Please note that the individual model components integrated in this project may have their own license requirements, please follow the original model license terms when using them. 