import torch
import os
import logging
import numpy as np
import string
import pandas as pd
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

# Imports for Native/Custom model
from olmo.config import ModelConfig
from olmo.model import Molmo as NativeMolmoModel
from olmo.data.model_preprocessor import MultiModalPreprocessor
from olmo.data.data_formatter import DataFormatter

# Helper for Native Model
class SafeDataFormatter(DataFormatter):
    def get_system_prompt(self, style, for_inference, messages, rng=None):
        if style is None:
            style = "User"
        return super().get_system_prompt(style, for_inference, messages, rng)

TYPE_PROMPTS = {
    'Y/N': 'vqa2:',
    'VQA': 'vqa2:',
    'MCQ': 'a_okvqa_mc:',
}

DATASET_PROMPTS = {
    'AI2D_TEST': 'ai2_diagram:',
    'AI2D_TEST_NO_MASK': 'ai2_diagram:',
    'COCO_VAL': 'coco_captioning:',
    'ChartQA_TEST': 'chart_qa:',
    'ChartQA_VAL': 'chart_qa:',
    'DocVQA_VAL': 'doc_qa:',
    'DocVQA_TEST': 'doc_qa:',
    'InfoVQA_TEST': 'info_qa:',
    'InfoVQA_VAL': 'info_qa:',
    'OCRVQA_TEST': 'ocr_vqa:',
    'OCRVQA_TESTCORE': 'ocr_vqa:',
    'ScienceQA_VAL': 'science_qa:',
    'ScienceQA_TEST': 'science_qa:',
    'TableVQABench': 'tabwmp_da:',
    'TextVQA_VAL': 'text_vqa:'
}

class molmo(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='allenai/Molmo-7B-D-0924', **kwargs):
        # Handle file path input
        if os.path.isfile(model_path):
            logging.warning(f"File path provided: {model_path}. Using parent directory.")
            model_path = os.path.dirname(model_path)
            
        self.model_path = model_path
        # 72B uses device_map="auto" across 8 GPUs, leaving ~500MB free per GPU after model weights.
        # max_crops=36 (~860MB/head) OOMs; max_crops=12 (~95MB/head) fits safely within the limit.
        default_max_crops = 12 if '72b' in model_path.lower() else 36
        self.max_crops = kwargs.get('max_crops', default_max_crops)
        self.kwargs = kwargs

        # Check for Native/Fine-tuned Checkpoint (config.yaml + model.pt)
        config_path = os.path.join(model_path, "config.yaml")
        checkpoint_path = os.path.join(model_path, "model.pt")

        if os.path.exists(config_path) and os.path.exists(checkpoint_path):
            self.is_native = True
            logging.info(f"[Molmo] Detected Native Checkpoint at {model_path}. Loading with olmo library...")
            self._init_native_model(model_path)
        else:
            self.is_native = False
            logging.info(f"[Molmo] Loading HF model from {model_path}...")
            self._init_hf_model(model_path, **kwargs)

    def _init_native_model(self, model_path):
        # Prevent PyTorch UnpicklingError
        if hasattr(torch, 'load'):
            _original_load = torch.load
            def _unsafe_load_wrapper(*args, **kwargs):
                if 'weights_only' not in kwargs: kwargs['weights_only'] = False
                return _original_load(*args, **kwargs)
            torch.load = _unsafe_load_wrapper

        config_path = os.path.join(model_path, "config.yaml")
        checkpoint_path = os.path.join(model_path, "model.pt")

        cfg = ModelConfig.load(config_path, key="model", validate_paths=False)
        cfg.init_device = "cpu"

        self.model = NativeMolmoModel(cfg)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device, dtype=torch.bfloat16).eval()

        self.tokenizer = cfg.get_tokenizer()
        v_cfg = cfg.vision_backbone
        h, w = cfg.llm_patches_per_crop()
        
        image_padding_mask = 2 if cfg.fix_image_padding else (1 if cfg.image_padding_embed else None)

        self.formatter = SafeDataFormatter(
            prompt_templates=cfg.prompt_type,
            message_format=cfg.message_formatting,
            system_prompt=cfg.system_prompt_kind,
            always_start_with_space=cfg.always_start_with_space,
            default_inference_len=cfg.default_inference_len
        )

        self.preprocessor = MultiModalPreprocessor(
            tokenizer=self.tokenizer,
            normalize=str(v_cfg.image_model_type),
            crop_mode=cfg.crop_mode,
            max_crops=cfg.max_crops,
            overlap_margins=cfg.overlap_margins,
            resize=v_cfg.resize_mode,
            use_col_tokens=cfg.use_col_tokens,
            base_image_input_size=v_cfg.image_default_input_size,
            image_pooling_w=cfg.image_pooling_w,
            image_pooling_h=cfg.image_pooling_h,
            image_token_length_w=w,
            image_token_length_h=h,
            image_patch_size=v_cfg.image_patch_size,
            image_padding_mask=image_padding_mask,
            pad_value=cfg.pad_value,
            loss_token_weighting=cfg.multi_annotation_weighting,
        )

    def _init_hf_model(self, model_path, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import einops
        except Exception as e:
            logging.critical('Please install transformer and einops before using molmo.')
            raise e

        if '72b' not in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='cuda')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model_name = model_path

    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) in ['Y/N', 'MCQ', 'VQA']:
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        prefix = None
        if dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            prompt = self.build_prompt_mcq_vqa(line)
        elif dataset in ['MathVista_MINI']:
            prompt = self.build_prompt_mathvista(line)
        elif dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            prompt = self.build_prompt_ai2d(line)
        elif dataset is not None and listinstr(list(DATASET_PROMPTS.keys()), dataset):
            prefix = DATASET_PROMPTS[dataset]
            prompt = self.build_prompt_vqa(line, prefix)
        elif dataset is not None and listinstr(['MCQ'], DATASET_TYPE(dataset)):
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            message = MMMUDataset.split_MMMU(message)
        return message

    def build_prompt_mathvista(self, line):
        if line['question_type'] == 'multi_choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_ai2d(self, line):
        def option_is_abc(line):
            for cand in string.ascii_uppercase:
                if cand in line and not pd.isna(line[cand]):
                    if not line[cand].strip().isalpha() or len(line[cand].strip()) > 1:
                        return False
            return True

        if line['abcLabel'] and option_is_abc(line):
            prompt = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                prompt += f'\n{item}'
            prompt = f"ai2_diagram_no_letter: {prompt}"
        else:
            prompt = self.build_prompt_multiple_choice(line, prefix='ai2_diagram:')
        return prompt

    def build_prompt_mcq_vqa(self, line):
        if line['question_type'] == 'multiple-choice':
            prompt = self.build_prompt_multiple_choice(line)
        else:
            prompt = self.build_prompt_vqa(line)
        return prompt

    def build_prompt_multiple_choice(self, line, prefix=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}: {item}'
        if prefix is None:
            prompt = f"{TYPE_PROMPTS['MCQ']} {question}"
        else:
            prompt = f"{prefix} {question}"
        return prompt

    def build_prompt_vqa(self, line, prefix=None):
        question = line['question']
        if prefix is None:
            prompt = f"{TYPE_PROMPTS['VQA']} {question}"
        else:
            prompt = f"{prefix} {question}"
        return prompt

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.is_native:
            # --- Native Molmo Inference (Custom Code) ---
            example = {
                "messages": [prompt],
                "image": image
            }
            
            # 1. Formatter
            messages, _ = self.formatter(example, is_training=False, for_inference=True, rng=np.random)
            
            # 2. Image -> Numpy Array
            image_np = np.array(image)
            
            # 3. Preprocessor
            batch = self.preprocessor(image_np, messages, is_training=False, require_image_features=True)
            
            if 'input_ids' not in batch and 'input_tokens' in batch:
                batch['input_ids'] = batch['input_tokens']
            
            def to_tensor(x):
                if isinstance(x, np.ndarray):
                    return torch.from_numpy(x)
                return x

            input_ids = to_tensor(batch['input_ids']).unsqueeze(0).to(self.device)
            if input_ids.dtype not in [torch.long, torch.int64]:
                input_ids = input_ids.long()
                
            images_tensor = to_tensor(batch['images']).unsqueeze(0).to(self.device).to(dtype=torch.bfloat16)
            image_masks = to_tensor(batch['image_masks']).unsqueeze(0).to(self.device).to(dtype=torch.bfloat16)
            image_input_idx = to_tensor(batch['image_input_idx']).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    output = self.model.generate(
                        input_ids=input_ids,
                        images=images_tensor,
                        image_masks=image_masks,
                        image_input_idx=image_input_idx,
                        max_steps=200,
                        beam_size=1,
                    )
            
            generated_ids = output.token_ids[0, 0]
            generated_text = self.tokenizer.decode(generated_ids.tolist(), truncate_at_eos=True).strip()

        else:
            # --- HF Molmo Inference (Original Code) ---
            from transformers import GenerationConfig
            
            # process the image and text
            inputs = self.processor.process(
                images=[image],
                text=prompt,
                images_kwargs={
                    "max_crops": self.max_crops
                }
            )

            # move inputs to the correct device and make a batch of size 1
            inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

            # 72B uses device_map="auto" across multiple GPUs.
            # autocast alone doesn't guarantee float32→bfloat16 conversion for patch_embedding in this case,
            # so cast explicitly.
            if '72b' in self.model_name.lower():
                inputs = {k: v.to(dtype=torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}

            # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                output = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=self.processor.tokenizer
                )

            # only get generated tokens; decode them to text
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # AI2D Post-processing (Common)
        if dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            if 'ai2_diagram_no_letter' in prompt:
                options = prompt.split('\n')[1:]
                try:
                    answer = options.index(generated_text)
                    generated_text = chr(answer + ord('A'))
                except ValueError:
                    pass

        return generated_text