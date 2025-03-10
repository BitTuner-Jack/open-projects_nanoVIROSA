# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np
import PIL.Image
import subprocess
import tempfile
import os
from torchvision.ops import roi_align
from transformers.models.owlvit.modeling_owlvit import OwlViTForObjectDetection
from transformers.models.owlvit.processing_owlvit import OwlViTProcessor
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from .image_preprocessor import ImagePreprocessor

__all__ = [
    "OwlPredictor",
    "OwlEncodeTextOutput",
    "OwlEncodeImageOutput",
    "OwlDecodeOutput"
]


def _owl_center_to_corners_format_torch(bboxes_center):
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        [
            (center_x - 0.5 * width), 
            (center_y - 0.5 * height), 
            (center_x + 0.5 * width), 
            (center_y + 0.5 * height)
        ],
        dim=-1,
    )
    return bbox_corners


def _owl_get_image_size(hf_name: str):

    image_sizes = {
        "google/owlvit-base-patch32": 768,
        "google/owlvit-base-patch16": 768,
        "google/owlvit-large-patch14": 840,
        "google/owlv2-base-patch16-ensemble":960
    }

    return image_sizes[hf_name]


def _owl_get_patch_size(hf_name: str):

    patch_sizes = {
        "google/owlvit-base-patch32": 32,
        "google/owlvit-base-patch16": 16,
        "google/owlvit-large-patch14": 14,
        "google/owlv2-base-patch16-ensemble": 16
    }

    return patch_sizes[hf_name]


# This function is modified from https://github.com/huggingface/transformers/blob/e8fdd7875def7be59e2c9b823705fbf003163ea0/src/transformers/models/owlvit/modeling_owlvit.py#L1333
# Copyright 2022 Google AI and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
def _owl_normalize_grid_corner_coordinates(num_patches_per_side):
    box_coordinates = np.stack(
        np.meshgrid(np.arange(1, num_patches_per_side + 1), np.arange(1, num_patches_per_side + 1)), axis=-1
    ).astype(np.float32)
    box_coordinates /= np.array([num_patches_per_side, num_patches_per_side], np.float32)

    box_coordinates = box_coordinates.reshape(
        box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
    )
    box_coordinates = torch.from_numpy(box_coordinates)

    return box_coordinates


# This function is modified from https://github.com/huggingface/transformers/blob/e8fdd7875def7be59e2c9b823705fbf003163ea0/src/transformers/models/owlvit/modeling_owlvit.py#L1354
# Copyright 2022 Google AI and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
def _owl_compute_box_bias(num_patches_per_side):
    box_coordinates = _owl_normalize_grid_corner_coordinates(num_patches_per_side)
    box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

    box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

    box_size = torch.full_like(box_coord_bias, 1.0 / num_patches_per_side)
    box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

    box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)

    return box_bias


def _owl_box_roi_to_box_global(boxes, rois):
    x0y0 = rois[..., :2]
    x1y1 = rois[..., 2:]
    wh = (x1y1 - x0y0).repeat(1, 1, 2)
    x0y0 = x0y0.repeat(1, 1, 2)
    return (boxes * wh) + x0y0


@dataclass
class OwlEncodeTextOutput:
    text_embeds: torch.Tensor

    def slice(self, start_index, end_index):
        return OwlEncodeTextOutput(
            text_embeds=self.text_embeds[start_index:end_index]
        )


@dataclass
class OwlEncodeImageOutput:
    image_embeds: torch.Tensor
    image_class_embeds: torch.Tensor
    logit_shift: torch.Tensor
    logit_scale: torch.Tensor
    pred_boxes: torch.Tensor


@dataclass
class OwlDecodeOutput:
    labels: torch.Tensor
    scores: torch.Tensor
    boxes: torch.Tensor
    input_indices: torch.Tensor


class OwlPredictor(torch.nn.Module):
    
    def __init__(self,
            model_name: str = "google/owlvit-base-patch32",
            device: str = "cuda",
            image_encoder_engine: Optional[str] = None,
            image_encoder_engine_max_batch_size: int = 1,
            image_preprocessor: Optional[ImagePreprocessor] = None
        ):

        super().__init__()

        self.image_size = _owl_get_image_size(model_name)
        self.device = device
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.patch_size = _owl_get_patch_size(model_name)
        self.num_patches_per_side = self.image_size // self.patch_size
        self.box_bias = _owl_compute_box_bias(self.num_patches_per_side).to(self.device)
        self.num_patches = (self.num_patches_per_side)**2
        self.mesh_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0., 1., self.image_size),
                torch.linspace(0., 1., self.image_size),
                indexing='ij'
            )
        ).to(self.device).float()
        self.image_encoder_engine = None
        if image_encoder_engine is not None:
            image_encoder_engine = OwlPredictor.load_image_encoder_engine(image_encoder_engine, image_encoder_engine_max_batch_size)
        self.image_encoder_engine = image_encoder_engine
        self.image_preprocessor = image_preprocessor.to(self.device).eval() if image_preprocessor else ImagePreprocessor().to(self.device).eval()

    def get_num_patches(self):
        return self.num_patches

    def get_device(self):
        return self.device
        
    def get_image_size(self):
        return (self.image_size, self.image_size)
    
    def encode_text(self, text: List[str]) -> OwlEncodeTextOutput:
        # 假设我们输入的文本列表 text 包含 B 个文本句子，每个句子的最大 token 数量（经过分词和填充后）为 S，
        # 文本模型的隐藏层维度为 D（通常与图像模型的隐藏层维度相同，例如 768）。

        text_input = self.processor(text=text, return_tensors="pt") # text_input is dict.
        input_ids = text_input['input_ids'].to(self.device) # (B, S)，其中 B 是批次大小，S 是序列长度（最大 token 数量）。每个元素是 token 的 ID。
        attention_mask = text_input['attention_mask'].to(self.device) # (B, S)，用于指示哪些 token 是实际文本，哪些是填充 token。1 表示实际文本，0 表示填充。
        text_outputs = self.model.owlvit.text_model(input_ids, attention_mask)
        # text_outputs 通常是一个元组，包含多个输出，其中：
        # text_outputs[0] 通常是所有 token 的隐藏状态，形状为 (B, S, D)
        # text_outputs[1] 通常是 pooled output 或 CLS token 的隐藏状态，形状为 (B, D), 文本嵌入。
        text_embeds = text_outputs[1]
        text_embeds = self.model.owlvit.text_projection(text_embeds) # (B, D)
        return OwlEncodeTextOutput(text_embeds=text_embeds)

    def encode_image_torch(self, image: torch.Tensor) -> OwlEncodeImageOutput:
        
        # image: [B, C, H, W], patch is P, then N = (H // P) * (W // P) patch nums
        vision_outputs = self.model.owlvit.vision_model(image)
        last_hidden_state = vision_outputs[0] # last_hidden_state includes image feat. [B, N+1, D(768)] ("1" class token)
        image_embeds = self.model.owlvit.vision_model.post_layernorm(last_hidden_state) # (B, N + 1, D)
        class_token_out = image_embeds[:, :1, :]  # (B, 1, D)
        image_embeds = image_embeds[:, 1:, :] * class_token_out  # image * class, (B, N, D)
        image_embeds = self.model.layer_norm(image_embeds)  # (B, N, D)

        # Box Head
        pred_boxes = self.model.box_head(image_embeds)  # (B, N, 4)，其中 4 表示 (x_center, y_center, width, height)
        pred_boxes += self.box_bias  
        pred_boxes = torch.sigmoid(pred_boxes)
        pred_boxes = _owl_center_to_corners_format_torch(pred_boxes)  # (B, N, 4)，每个 patch 预测的边界框坐标 (x_min, y_min, x_max, y_max)

        # Class Head
        image_class_embeds = self.model.class_head.dense0(image_embeds)  # (B, N, D)
        logit_shift = self.model.class_head.logit_shift(image_embeds)  # (B, N)
        logit_scale = self.model.class_head.logit_scale(image_embeds)  # (B, N)
        logit_scale = self.model.class_head.elu(logit_scale) + 1

        output = OwlEncodeImageOutput(
            image_embeds=image_embeds,
            image_class_embeds=image_class_embeds,
            logit_shift=logit_shift,
            logit_scale=logit_scale,
            pred_boxes=pred_boxes
        )

        return output
    
    def encode_image_trt(self, image: torch.Tensor) -> OwlEncodeImageOutput:
        return self.image_encoder_engine(image)

    def encode_image(self, image: torch.Tensor) -> OwlEncodeImageOutput:
        if self.image_encoder_engine is not None:
            return self.encode_image_trt(image)
        else:
            return self.encode_image_torch(image)

    def extract_rois(self, image: torch.Tensor, rois: torch.Tensor, pad_square: bool = True, padding_scale: float = 1.0):
        if len(rois) == 0:
            return torch.empty(
                (0, image.shape[1], self.image_size, self.image_size),
                dtype=image.dtype,
                device=image.device
            )
        if pad_square:
            # pad square
            w = padding_scale * (rois[..., 2] - rois[..., 0]) / 2
            h = padding_scale * (rois[..., 3] - rois[..., 1]) / 2
            cx = (rois[..., 0] + rois[..., 2]) / 2
            cy = (rois[..., 1] + rois[..., 3]) / 2
            s = torch.max(w, h)
            rois = torch.stack([cx-s, cy-s, cx+s, cy+s], dim=-1)

            # compute mask
            pad_x = (s - w) / (2 * s)
            pad_y = (s - h) / (2 * s)
            mask_x = (self.mesh_grid[1][None, ...] > pad_x[..., None, None]) & (self.mesh_grid[1][None, ...] < (1. - pad_x[..., None, None]))
            mask_y = (self.mesh_grid[0][None, ...] > pad_y[..., None, None]) & (self.mesh_grid[0][None, ...] < (1. - pad_y[..., None, None]))
            mask = (mask_x & mask_y)

        # extract rois
        roi_images = roi_align(image, [rois], output_size=self.get_image_size())

        # mask rois
        if pad_square:
            roi_images = (roi_images * mask[:, None, :, :])

        return roi_images, rois
    
    def encode_rois(self, image: torch.Tensor, rois: torch.Tensor, pad_square: bool = True, padding_scale: float=1.0):
        # with torch_timeit_sync("extract rois"):
        roi_images, rois = self.extract_rois(image, rois, pad_square, padding_scale)
        # with torch_timeit_sync("encode images"):
        output = self.encode_image(roi_images)
        pred_boxes = _owl_box_roi_to_box_global(output.pred_boxes, rois[:, None, :])
        output.pred_boxes = pred_boxes
        return output

    def decode(self, 
            image_output: OwlEncodeImageOutput, 
            text_output: OwlEncodeTextOutput,
            threshold: Union[int, float, List[Union[int, float]]] = 0.1,
        ) -> OwlDecodeOutput:

        # B: 批次大小（输入图像的数量）。
        # N: 每个图像中的 patch 数量。
        # Q: 文本 query 的数量（即要检测的物体类别的数量）。
        # D: 嵌入向量的维度（例如 768）。

        if isinstance(threshold, (int, float)):
            threshold = [threshold] * len(text_output.text_embeds) #apply single threshold to all labels 

        num_input_images = image_output.image_class_embeds.shape[0]

        image_class_embeds = image_output.image_class_embeds  # (B, N, D)
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = text_output.text_embeds  # (Q, D)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # (B, N, Q) logits[b, n, q] 表示第 b 张图像的第 n 个 patch 与第 q 个文本 query 的相似度。
        logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        logits = (logits + image_output.logit_shift) * image_output.logit_scale    # (B, N, Q)
        
        scores_sigmoid = torch.sigmoid(logits)  # (B, N, Q)
        scores_max = scores_sigmoid.max(dim=-1)  
        labels = scores_max.indices  # (B, N)，表示每个 patch 的最大 score。
        scores = scores_max.values   # (B, N)，表示每个 patch 对应的最大 score 的 query 索引（即标签）。
        masks = []
        for i, thresh in enumerate(threshold):
            label_mask = labels == i  # (B, N)，表示哪些 patch 的预测标签是 i 
            score_mask = scores > thresh  # (B, N)，表示哪些 patch 的 score 大于当前阈值。
            obj_mask = torch.logical_and(label_mask,score_mask)  # (B, N)，表示哪些 patch 既是类别 i，又超过了阈值。
            masks.append(obj_mask) 
        
        mask = masks[0]
        for mask_t in masks[1:]:
            # 将所有类别的 mask 进行逻辑或运算，得到最终的 mask，形状为 (B, N)。mask[b, n] 为 True 表示第 b 张图像的第 n 个 patch 被检测到。
            mask = torch.logical_or(mask, mask_t)

        input_indices = torch.arange(0, num_input_images, dtype=labels.dtype, device=labels.device) # 创建输入图像的索引，形状为 (B,)
        input_indices = input_indices[:, None].repeat(1, self.num_patches) # 将索引扩展为 (B, N)，与 labels 和 scores 的形状匹配

        # 都会被 mask 索引，只保留被检测到的 patch 的信息。它们的形状会变为 (M,)、(M,)、(M, 4) 和 (M,)，其中 M 是被检测到的 patch 的总数。
        return OwlDecodeOutput(
            labels=labels[mask],
            scores=scores[mask],
            boxes=image_output.pred_boxes[mask],
            input_indices=input_indices[mask]
        )

    @staticmethod
    def get_image_encoder_input_names():
        return ["image"]

    @staticmethod
    def get_image_encoder_output_names():
        names = [
            "image_embeds",
            "image_class_embeds",
            "logit_shift",
            "logit_scale",
            "pred_boxes"
        ]
        return names


    def export_image_encoder_onnx(self, 
            output_path: str,
            use_dynamic_axes: bool = True,
            batch_size: int = 1,
            onnx_opset=17
        ):
        
        class TempModule(torch.nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            def forward(self, image):
                output = self.parent.encode_image_torch(image)
                return (
                    output.image_embeds,
                    output.image_class_embeds,
                    output.logit_shift,
                    output.logit_scale,
                    output.pred_boxes
                )

        data = torch.randn(batch_size, 3, self.image_size, self.image_size).to(self.device)

        if use_dynamic_axes:
            dynamic_axes =  {
                "image": {0: "batch"},
                "image_embeds": {0: "batch"},
                "image_class_embeds": {0: "batch"},
                "logit_shift": {0: "batch"},
                "logit_scale": {0: "batch"},
                "pred_boxes": {0: "batch"}       
            }
        else:
            dynamic_axes = {}

        model = TempModule(self)

        torch.onnx.export(
            model, 
            data, 
            output_path, 
            input_names=self.get_image_encoder_input_names(), 
            output_names=self.get_image_encoder_output_names(),
            dynamic_axes=dynamic_axes,
            opset_version=onnx_opset
        )
    
    @staticmethod
    def load_image_encoder_engine(engine_path: str, max_batch_size: int = 1):
        import tensorrt as trt
        from torch2trt import TRTModule

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        base_module = TRTModule(
            engine,
            input_names=OwlPredictor.get_image_encoder_input_names(),
            output_names=OwlPredictor.get_image_encoder_output_names()
        )

        class Wrapper(torch.nn.Module):
            def __init__(self, base_module: TRTModule, max_batch_size: int):
                super().__init__()
                self.base_module = base_module
                self.max_batch_size = max_batch_size

            @torch.no_grad()
            def forward(self, image):

                b = image.shape[0]

                results = []

                for start_index in range(0, b, self.max_batch_size):
                    end_index = min(b, start_index + self.max_batch_size)
                    image_slice = image[start_index:end_index]
                    # with torch_timeit_sync("run_engine"):
                    output = self.base_module(image_slice)
                    results.append(
                        output
                    )

                return OwlEncodeImageOutput(
                    image_embeds=torch.cat([r[0] for r in results], dim=0),
                    image_class_embeds=torch.cat([r[1] for r in results], dim=0),
                    logit_shift=torch.cat([r[2] for r in results], dim=0),
                    logit_scale=torch.cat([r[3] for r in results], dim=0),
                    pred_boxes=torch.cat([r[4] for r in results], dim=0)
                )

        image_encoder = Wrapper(base_module, max_batch_size)

        return image_encoder

    def build_image_encoder_engine(self, 
            engine_path: str, 
            max_batch_size: int = 1, 
            fp16_mode = True, 
            onnx_path: Optional[str] = None,
            onnx_opset: int = 17
        ):

        if onnx_path is None:
            onnx_dir = tempfile.mkdtemp()
            onnx_path = os.path.join(onnx_dir, "image_encoder.onnx")
            self.export_image_encoder_onnx(onnx_path, onnx_opset=onnx_opset)

        args = ["/usr/src/tensorrt/bin/trtexec"]
    
        args.append(f"--onnx={onnx_path}")
        args.append(f"--saveEngine={engine_path}")

        if fp16_mode:
            args += ["--fp16"]

        args += [f"--shapes=image:1x3x{self.image_size}x{self.image_size}"]

        subprocess.call(args)

        return self.load_image_encoder_engine(engine_path, max_batch_size)

    def predict(self, 
            image: PIL.Image, 
            text: List[str], 
            text_encodings: Optional[OwlEncodeTextOutput],
            threshold: Union[int, float, List[Union[int, float]]] = 0.1,
            pad_square: bool = True,
            
        ) -> OwlDecodeOutput:

        image_tensor = self.image_preprocessor.preprocess_pil_image(image)

        if text_encodings is None:
            text_encodings = self.encode_text(text)

        rois = torch.tensor([[0, 0, image.width, image.height]], dtype=image_tensor.dtype, device=image_tensor.device)

        image_encodings = self.encode_rois(image_tensor, rois, pad_square=pad_square)

        return self.decode(image_encodings, text_encodings, threshold)

