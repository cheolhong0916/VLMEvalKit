import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class Molmo2Chat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='allenai/Molmo2-8B', **kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_path = model_path

        if '72b' in model_path.lower():
            device_map = 'auto'
        else:
            device_map = 'auto'

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map=device_map,
        )
        self.model.eval()
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        # Build messages dict with PIL images
        content = []
        images = []

        for item in message:
            if item['type'] == 'text':
                content.append(dict(type='text', text=item['value']))
            elif item['type'] == 'image':
                img = Image.open(item['value'])
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                content.append(dict(type='image', image=img))

        messages = [{'role': 'user', 'content': content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text.strip()
