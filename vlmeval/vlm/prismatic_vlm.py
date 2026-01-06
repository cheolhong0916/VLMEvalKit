# vlmeval/vlm/prismatic_vlm.py

import torch
from PIL import Image
import os.path as osp
import warnings
import logging
from .base import BaseModel  # Import from VLMEvalKit's base
from ..smp import isimg      # Import from VLMEvalKit's smp utils
import sys

prismatic_project_path = "/data/shared/Qwen/prismatic-vlms" 
if prismatic_project_path not in sys.path:
    sys.path.append(prismatic_project_path)

try:
    # Try to import the prismatic library
    from prismatic import load
except: pass
# except ImportError:
#     logging.error(
#         "Failed to import prismatic. "
#         "Please install it first: pip install git+https://github.com/TRI-ML/prismatic-vlms"
#     )
#     # Set to None to handle in __init__
#     load = None

class PrismaticVLM(BaseModel):
    """
    Wrapper for the PrismaticVLM models (https://github.com/TRI-ML/prismatic-vlms).

    This wrapper loads a fine-tuned model from a local checkpoint directory.
    The directory must contain 'config.json' and 'checkpoints/latest-checkpoint.pt'.
    """

    INSTALL_REQ = True  # Requires external pip install
    INTERLEAVE = True   # Handles interleaved image/text

    def __init__(self, model_path: str, hf_token: str = None, **kwargs):
        """
        Initialize the PrismaticVLM model.

        Args:
            model_path (str): Path to the local fine-tuned model directory.
            hf_token (str, optional): HuggingFace token for gated base models (e.g., Llama-2).
            **kwargs: Additional generation parameters.
        """
        super().__init__()
        
        if load is None:
            raise ImportError(
                "prismatic library not found. "
                "Please install it: pip install git+https://github.com/TRI-ML/prismatic-vlms"
            )

        # Check if the model path is a valid directory
        if not osp.isdir(model_path):
            raise ValueError(
                f"model_path must be a local directory containing the fine-tuned model, but got: {model_path}"
            )

        # Load the VLM using the prismatic.load() function
        # This handles loading the vision backbone, LLM, and projector weights
        try:
            self.vlm = load(model_path, hf_token=hf_token)
        except Exception as e:
            logging.error(f"Failed to load PrismaticVLM from path: {model_path}. Error: {e}")
            raise
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vlm.to(self.device)
        self.vlm.eval()
        
        # Set default generation kwargs
        # VLMEvalKit evaluations typically prefer deterministic output
        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=512,  # Default, can be overridden by user
            min_length=1,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"PrismaticVLM wrapper initialized with generation config: {self.kwargs}")

    # [!! MODIFIED FUNCTION !!]
    # This function is now patched to handle both
    # (1) standard list-of-dicts format
    # (2) ERQA's non-standard list-of-strings format
    def _build_prompt_and_get_image(self, message):
        """
        Helper function to parse the VLMEvalKit message format,
        extract the image, and build the text prompt.
        Patched to handle non-standard list-of-strings format from ERQA.
        """
        image_path = None
        image = None
        prompt_text = ""
        
        if not isinstance(message, list) or len(message) == 0:
            logging.error("Invalid message format: not a list or empty.")
            return None, ""

        # Check 1: Standard VLMEvalKit format (list of dicts)
        # We check this by seeing if all elements are dictionaries
        is_standard_format = all(isinstance(x, dict) for x in message)

        if is_standard_format:
            try:
                # --- Find Image (Standard) ---
                if 'role' in message[0]:
                     for turn in reversed(message):
                        for item in reversed(turn['content']):
                            if item['type'] == 'image':
                                image_path = item['value']
                                break
                        if image_path:
                            break
                else:
                     for item in reversed(message):
                        if item['type'] == 'image':
                            image_path = item['value']
                            break
                
                # --- Build Prompt (Standard) ---
                prompt_builder = self.vlm.get_prompt_builder()
                if 'role' in message[0]:
                    for i, turn in enumerate(message):
                        role = "human" if turn["role"] == "user" else "gpt"
                        turn_text = ""
                        for item in turn["content"]:
                            if item["type"] == "text":
                                turn_text += item["value"]
                        prompt_builder.add_turn(role=role, message=turn_text)
                else:
                    turn_text = ""
                    for item in message:
                        if item["type"] == "text":
                            turn_text += item["value"]
                    prompt_builder.add_turn(role="human", message=turn_text)
                
                prompt_text = prompt_builder.get_prompt()

            except (KeyError, TypeError) as e:
                logging.error(f"Standard message format parsing failed: {e}")
                return None, ""

        # Check 2: ERQA's broken format (list of strings/mixed)
        else:
            # logging.warning("Detected non-standard message format, attempting ERQA compatibility patch.")
            
            image_paths = []
            text_parts = []
            
            for item in message:
                # Handle strings (check if it's an image path or text)
                if isinstance(item, str):
                    if isimg(item): # isimg() checks file extension
                        image_paths.append(item)
                    else:
                        text_parts.append(item)
                # Handle dicts mixed in (just in case)
                elif isinstance(item, dict) and item.get('type') == 'image':
                    image_paths.append(item.get('value'))
                elif isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('value'))
            
            if image_paths:
                image_path = image_paths[-1] # Use the last image
            
            # Build the prompt text
            prompt_builder = self.vlm.get_prompt_builder()
            full_text = " ".join(text_parts) # Just concatenate all text parts
            prompt_builder.add_turn(role="human", message=full_text)
            prompt_text = prompt_builder.get_prompt()

        # --- Post-processing (same for both) ---
        if not image_path:
            # This will catch text-only questions (which ERQA shouldn't have, but good to keep)
            logging.warning("No image found in message for PrismaticVLM.")
            image = None
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logging.error(f"Failed to open image at path: {image_path}. Error: {e}")
                image = None
        
        return image, prompt_text

    def generate(self, message, dataset=None):
        """
        Handle single-turn generation requests.
        """
        # Use the (now patched) helper to parse message and build prompt
        image, prompt_text = self._build_prompt_and_get_image(message)
        
        if image is None:
            logging.error("PrismaticVLM generate failed: No image provided.")
            return "Error: No image provided."
            
        try:
            # Call the core generate function from prismatic.py
            generated_text = self.vlm.generate(
                image,
                prompt_text,
                **self.kwargs
            )
        except Exception as e:
            logging.error(f"Error during PrismaticVLM generation: {e}")
            return f"Error: {e}"
        
        return generated_text

    def chat(self, message, dataset=None):
        """
        Handle multi-turn chat requests.
        
        PrismaticVLM's generate function handles multi-turn prompts
        if the prompt_text is formatted correctly by the prompt_builder.
        """
        # The (now patched) helper works for multi-turn messages
        image, prompt_text = self._build_prompt_and_get_image(message)
        
        if image is None:
            logging.error("PrismaticVLM chat failed: No image provided.")
            return "Error: No image provided."
            
        try:
            # Call the same core generate function
            generated_text = self.vlm.generate(
                image,
                prompt_text,
                **self.kwargs
            )
        except Exception as e:
            logging.error(f"Error during PrismaticVLM chat generation: {e}")
            return f"Error: {e}"
        
        return generated_text