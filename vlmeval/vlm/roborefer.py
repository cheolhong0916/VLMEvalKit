# vlmeval/vlm/roborefer.py

import torch
import os
import os.path as osp
import uuid
import time
from ..smp import *
from .base import BaseModel

import sys
sys.path.append('/data/shared/Qwen/RoboRefer')
sys.path.append('/data/shared/Qwen/RoboRefer/API')

try:
    import llava
    from llava import conversation as clib
    from llava.media import Image, Depth
except ImportError:
    raise ImportError("llava is not loaded.")

try:
    import cv2
    import numpy as np
    from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    DepthAnythingV2 = None

class RoboRefer(BaseModel):
    """
    Direct integration of RoboRefer model into VLMEval-Kit.
    This version supports toggling depth processing on or off.
    """

    INTERLEAVE = True 

    def __init__(self,
                 vlm_model_path='',
                 depth_model_path=None,
                 depth_encoder='vitl',
                 enable_depth=False,
                 **kwargs):
        """
        Initializes the RoboRefer model. Conditionally initializes the Depth Anything model.

        Args:
            vlm_model_path (str): Path to the RoboRefer VLM model weights.
            depth_model_path (str, optional): Path to the Depth Anything V2 weights. Required if enable_depth is True.
            depth_encoder (str): Encoder type for the depth model.
            enable_depth (bool): Flag to enable depth processing. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.enable_depth = enable_depth
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vlm_model_path = vlm_model_path
        self.model_name = vlm_model_path.split('/')[-1] if vlm_model_path else 'RoboRefer'

        # Raw output logging
        self.work_dir = None
        self.raw_log_file = None
        self._log_counter = 0
        
        if self.enable_depth and DepthAnythingV2 is None:
            raise ImportError("Depth Anything V2 requirements are not met, but enable_depth is True. Please install opencv-python, numpy and clone the Depth_Anything_V2 repository.")

        if self.enable_depth and not depth_model_path:
            raise ValueError("`depth_model_path` must be provided when `enable_depth` is True.")
            
        self.vlm_model, self.depth_model = self._init_models(
            vlm_model_path, depth_model_path, depth_encoder
        )
        clib.default_conversation = clib.conv_templates['auto'].copy()

    def set_work_dir(self, work_dir, dataset_name=None):
        """Set the working directory for saving raw output logs."""
        self.work_dir = work_dir
        self._log_counter = 0  # Reset counter for each dataset
        if work_dir:
            # Include dataset name in log filename to avoid mixing logs
            if dataset_name:
                self.raw_log_file = osp.join(work_dir, f'{self.model_name}_{dataset_name}_raw_output.log')
            else:
                self.raw_log_file = osp.join(work_dir, f'{self.model_name}_raw_output.log')
            # Create new log file for this dataset
            with open(self.raw_log_file, 'w') as f:
                f.write(f"=== RoboRefer Raw Output Log ===\n")
                f.write(f"Model: {self.vlm_model_path}\n")
                f.write(f"Dataset: {dataset_name or 'unknown'}\n")
                f.write(f"Depth enabled: {self.enable_depth}\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

    def _log_raw_output(self, prompt_info, response):
        """Log model input/output to file for debugging."""
        if not self.raw_log_file:
            return

        try:
            with open(self.raw_log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"[Entry {self._log_counter}]\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"\n--- INPUT PROMPT ---\n")
                f.write(prompt_info if prompt_info else "(empty)")
                f.write(f"\n\n--- MODEL RESPONSE ---\n")
                f.write(response if response else "(empty)")
                f.write(f"\n{'='*60}\n")
            self._log_counter += 1
        except Exception as e:
            logging.warning(f"Failed to write raw output log: {e}")

    def _init_models(self, vlm_model_path, depth_model_path, depth_encoder):
        print(f"Loading VLM model from: {vlm_model_path}")
        vlm_model = llava.load(vlm_model_path)
        
        depth_model = None
        if self.enable_depth:
            print(f"Loading Depth model from: {depth_model_path}")
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            depth_model = DepthAnythingV2(**model_configs[depth_encoder])
            depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu'))
            depth_model = depth_model.to(self.device).eval()
            
        return vlm_model, depth_model

    def _get_depth_image(self, image_path):
        raw_image = cv2.imread(image_path)
        depth = self.depth_model.infer_image(raw_image, input_size=518, device=self.device)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        
        temp_dir = 'tmp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"depth_{uuid.uuid4().hex}.png")
        cv2.imwrite(temp_path, depth)
        return temp_path
        
    def generate_inner(self, message, dataset=None):
        prompt = []
        temp_depth_files = []
        prompt_info_parts = []  # For logging

        for msg in message:
            if msg['type'] == 'text':
                prompt.append(msg['value'])
                prompt_info_parts.append(f"[TEXT]: {msg['value']}")
            elif msg['type'] == 'image':
                image_path = msg['value']
                prompt.append(Image(image_path))
                prompt_info_parts.append(f"[IMAGE]: {image_path}")

                if self.enable_depth:
                    depth_path = self._get_depth_image(image_path)
                    prompt.append(Depth(depth_path))
                    temp_depth_files.append(depth_path)
                    prompt_info_parts.append(f"[DEPTH]: {depth_path}")

        try:
            answer = self.vlm_model.generate_content(prompt)

            # Log raw output for debugging
            prompt_info = "\n".join(prompt_info_parts)
            self._log_raw_output(prompt_info, answer)
        finally:
            for file_path in temp_depth_files:
                if os.path.exists(file_path):
                    os.remove(file_path)

        return answer