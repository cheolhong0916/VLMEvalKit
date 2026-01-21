import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from huggingface_hub import snapshot_download
from .base import BaseModel

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    pass


class SpatialLadder(BaseModel):
    """
    Integration of SpatialLadder-3B model into VLMEval-Kit.
    Based on Qwen2.5-VL architecture.
    """

    INTERLEAVE = True

    def __init__(self, model_path='hongxingli/SpatialLadder-3B', **kwargs):
        assert model_path is not None
        
        # 1. Download Model if needed
        if not os.path.exists(model_path):
            print(f"Downloading {model_path} from Hugging Face Hub...")
            try:
                model_path = snapshot_download(repo_id=model_path)
            except Exception as e:
                print(f"Warning: snapshot_download failed ({e}). Trying to load directly...")
        
        self.model_path = model_path
        print(f"Loading SpatialLadder model from: {self.model_path}")

        # 2. Load Model (following official usage)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        
        # 3. Load Processor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # 4. Default generation kwargs
        default_kwargs = dict(
            max_new_tokens=128,
            do_sample=False
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        """
        Generate response following official SpatialLadder usage pattern.
        
        Args:
            message: List of dicts with 'type' and 'value' keys
            dataset: Optional dataset name for adjusting generation parameters
        """
        # 1. Convert VLMEvalKit message format to Qwen format
        content_list = []
        for msg in message:
            if msg['type'] == 'image':
                content_list.append({'type': 'image', 'image': msg['value']})
            elif msg['type'] == 'text':
                content_list.append({'type': 'text', 'text': msg['value']})
        
        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

        # 2. Preparation for inference (following official code)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.model.device)

        # 3. Adjust generation parameters based on dataset
        gen_kwargs = self.kwargs.copy()
        if dataset and 'VQA' in dataset:
            gen_kwargs['max_new_tokens'] = max(gen_kwargs.get('max_new_tokens', 128), 32)

        # 4. Generation of the output (following official code)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # 5. Trim and decode (following official code)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]