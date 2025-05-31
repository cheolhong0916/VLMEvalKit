from ..smp import *
from .base import BaseModel

class RoboPoint(BaseModel):
    """
    Implementation of RoboPoint model for VLMEvalKit
    """

    INTERLEAVE = False  # RoboPoint doesn't support interleaved inputs
    
    def __init__(self, model_path=None, **kwargs):
        """
        Initialize the RoboPoint model.
        
        Args:
            model_path (str, optional): Path to the model weights. Defaults to 'wentao-yuan/robopoint-v1-vicuna-v1.5-13b'.
        """
        super().__init__()
        self.model_kwargs = self.load_robopoint_model(model_path)
        
    def load_robopoint_model(self, model_path=None):
        """
        Load the RoboPoint model and related components.
        
        Args:
            model_path (str, optional): Path to the model weights.
            
        Returns:
            dict: Model components including model, tokenizer, and image_processor.
        """
        # Robopoint-specific imports
        from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from robopoint.conversation import conv_templates
        from robopoint.model.builder import load_pretrained_model
        from robopoint.utils import disable_torch_init
        from robopoint.mm_utils import get_model_name_from_path

        # Disable torch initialization for faster loading
        disable_torch_init()

        # Use provided model path or default
        if model_path is None:
            model_path = 'wentao-yuan/robopoint-v1-vicuna-v1.5-13b'
        model_base = None  # Update if necessary

        # Load model name
        model_name = get_model_name_from_path(model_path)

        # Load tokenizer, model, image_processor, context_len
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

        # Prepare model kwargs
        model_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
        }

        return model_kwargs
    
    def build_prompt(self, line, dataset=None):
        """
        Build custom prompts for a specific dataset.
        
        Args:
            line: The raw input line.
            dataset: The name of the dataset.
            
        Returns:
            The built prompt message.
        """
        # For simplicity, returning a basic implementation
        # Customize this based on dataset-specific requirements
        if isinstance(line, int):
            line = self.data.iloc[line]
            
        question = line.get('question', '')
        if 'hint' in line and line['hint']:
            question = f"{line['hint']}\n{question}"
            
        return [
            dict(type='image', value=self.get_image_path(line)),
            dict(type='text', value=question)
        ]
    
    def generate_inner(self, message, dataset=None):
        """
        Generate output from the RoboPoint model.
        
        Args:
            message: List of dict containing the input message.
            dataset: The name of the dataset.
            
        Returns:
            str: The generated text response.
        """
        # Since RoboPoint doesn't support interleaved inputs, use the helper method
        prompt, image_path = self.message_to_promptimg(message, dataset)
        
        if image_path is None:
            # Return some error or fallback for text-only input
            return "RoboPoint requires image input."
        
        # Robopoint-specific imports
        import torch
        from PIL import Image
        from robopoint.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        )
        from robopoint.conversation import conv_templates
        from robopoint.mm_utils import tokenizer_image_token, process_images

        # Extract necessary components
        model = self.model_kwargs["model"]
        tokenizer = self.model_kwargs["tokenizer"]
        image_processor = self.model_kwargs["image_processor"]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Process the question
        if DEFAULT_IMAGE_TOKEN not in prompt:
            if getattr(model.config, 'mm_use_im_start_end', False):
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        # Conversation mode
        conv_mode = "llava_v1"  # Update if necessary
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize input
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)

        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().to(device)

        # Generate output
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Extract the assistant's response from the conversation
        if conv.roles[1] in outputs:
            outputs = outputs.split(conv.roles[1])[-1].strip()
            
        return outputs