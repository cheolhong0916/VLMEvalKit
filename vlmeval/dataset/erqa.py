import os
import re
import json
import base64
import io
import tempfile
from functools import partial
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


class ERQA(ImageBaseDataset):
    TYPE = "MCQ"
    ROBUST = True

    DATASET_URL = {
        "ERQA": "https://huggingface.co/datasets/ch-min/ERQA-tsv/resolve/main/erqa.tsv",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self, dataset):
        dataset = "ERQA"
        url = self.DATASET_URL[dataset]
        md5 = self.DATASET_MD5.get(dataset, None)
        return self.prepare_tsv(url, md5)

    def decode_base64_image(self, base64_str):
        """Decode base64 string to PIL Image"""
        try:
            img_data = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(img_data))
        except BaseException:
            return None

    def dump_image(self, line):
        """Override dump_image to handle ERQA's multiple image columns"""
        if isinstance(line, int):
            line = self.data.iloc[line]

        os.makedirs(self.img_root, exist_ok=True)

        # Collect all image data from multiple columns
        images = []
        for i in range(1, 17):  # image, image2, ..., image16
            col_name = 'image' if i == 1 else f'image{i}'
            if col_name in line and not pd.isna(
                    line[col_name]) and line[col_name].strip():
                images.append(line[col_name])

        if not images:
            return []

        tgt_paths = []
        for i, img_base64 in enumerate(images):
            try:
                # Use PIL Image approach like the working version
                pil_img = self.decode_base64_image(img_base64)
                if pil_img:
                    # Use /tmp/ path like the working version
                    tgt_path = f"/tmp/erqa_img_{line['index']}_{i}.png"
                    pil_img.save(tgt_path, 'PNG')
                    tgt_paths.append(tgt_path)
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue

        return tgt_paths

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        visual_indices = []

        # Parse visual_indices - use json.loads like the working version
        if 'visual_indices' in line and not pd.isna(line['visual_indices']):
            vi = line['visual_indices']
            if isinstance(vi, str):
                try:
                    visual_indices = json.loads(vi) if vi.strip() else []
                except BaseException:
                    visual_indices = []
            elif isinstance(vi, (list, np.ndarray)):
                visual_indices = list(vi)
            else:
                visual_indices = []

        # Get image paths using parent class method
        tgt_paths = self.dump_image(line)
        if not tgt_paths:
            return [dict(type='text', value=question)]

        # Build interleaved content based on visual_indices
        msgs = self.build_interleaved_content(
            question, tgt_paths, visual_indices)

        return msgs

    def build_interleaved_content(self, question, pil_images, visual_indices):
        """Build interleaved content based on visual_indices logic from eval_harness.py"""
        contents = []

        # Handle case where visual_indices is empty
        if len(visual_indices) == 0:
            for img in pil_images:
                contents.append(img)
            contents.append(question)
            return contents

        # Handle case where all indices are 0
        if all(idx == 0 for idx in visual_indices):
            image_index_pairs = list(zip(pil_images, visual_indices))
            for img, _ in image_index_pairs:
                contents.append(img)
            contents.append(question)
            return contents

        # General interleaved case
        image_index_pairs = list(zip(pil_images, visual_indices))
        image_index_pairs.sort(key=lambda x: x[1])

        last_pos = 0

        for img, idx in image_index_pairs:
            if idx == 0:
                contents.append(img)
            else:
                # Add text segment before this image
                if idx <= len(question):
                    text_segment = question[last_pos:idx]
                    if text_segment:
                        contents.append(text_segment)
                    contents.append(img)
                    last_pos = idx
                else:
                    # Index beyond question length, just append image
                    contents.append(img)

        # Add remaining text
        if last_pos < len(question):
            contents.append(question[last_pos:])

        # If no content was added, add full question and images
        if not contents:
            contents.append(question)
            for img, _ in image_index_pairs:
                contents.append(img)

        return contents

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)

        # Initialize counters
        total_examples = 0
        correct_examples = 0
        single_image_total = 0
        single_image_correct = 0
        multi_image_total = 0
        multi_image_correct = 0
        question_type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

        # Process each example
        for i in range(len(data)):
            line = data.iloc[i]
            index = line['index']

            # Get ground truth answer and prediction
            gt_answer = str(line['answer']).replace(".", "").strip().lower()
            prediction = str(
                line['prediction']).replace(
                ".",
                "").strip().lower()

            # Check correctness - exact match after normalization
            is_correct = gt_answer == prediction

            # Update counters
            total_examples += 1
            if is_correct:
                correct_examples += 1

            # Determine if single or multi-image based on original data
            original_line = self.data[self.data['index'] == index].iloc[0]

            # Count actual images from multiple columns
            num_images = 0
            for j in range(1, 17):  # image, image2, ..., image16
                col_name = 'image' if j == 1 else f'image{j}'
                if col_name in original_line and not pd.isna(
                        original_line[col_name]) and original_line[col_name].strip():
                    num_images += 1

            if num_images == 1:
                single_image_total += 1
                if is_correct:
                    single_image_correct += 1
            elif num_images > 1:
                multi_image_total += 1
                if is_correct:
                    multi_image_correct += 1

            # Track by question type if available
            if 'question_type' in line and not pd.isna(line['question_type']):
                q_type = str(line['question_type'])
                question_type_stats[q_type]['total'] += 1
                if is_correct:
                    question_type_stats[q_type]['correct'] += 1

        # Calculate results
        results = {
            'Correct': correct_examples,
            'Total': total_examples,
            'Accuracy': correct_examples / total_examples if total_examples > 0 else 0.0,
        }

        # Add accuracies
        if single_image_total > 0:
            results['Single_Image_Accuracy'] = single_image_correct / \
                single_image_total

        if multi_image_total > 0:
            results['Multi_Image_Accuracy'] = multi_image_correct / \
                multi_image_total

        for q_type, stats in question_type_stats.items():
            if stats['total'] > 0:
                results[f'{q_type}_Accuracy'] = stats['correct'] / \
                    stats['total']

        # Add total counts
        if single_image_total > 0:
            results['Single_Image_Correct'] = single_image_correct
            results['Single_Image_Total'] = single_image_total

        if multi_image_total > 0:
            results['Multi_Image_Correct'] = multi_image_correct
            results['Multi_Image_Total'] = multi_image_total

        for q_type, stats in question_type_stats.items():
            if stats['total'] > 0:
                results[f'{q_type}_Correct'] = stats['correct']
                results[f'{q_type}_Total'] = stats['total']

        # Save detailed results
        score_file = eval_file.replace('.xlsx', '_score.json')
        dump(results, score_file)

        return results
