import os
import re
import ast
import tempfile
from functools import partial

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


class RoboSpatial(ImageBaseDataset):
    TYPE = "VQA"
    # When ROBUST is True, if the models does not follow the format, all of
    # the response will be treated as answers.
    ROBUST = True

    DATASET_URL = {
        "RoboSpatial": "https://huggingface.co/datasets/ch-min/RoboSpatial-Home-tsv/resolve/main/RoboSpatial.tsv",
    }

    DATASET_MD5 = {
        "RoboSpatial": None  # Add MD5 hash for integrity check (optional)
        # To get MD5: md5sum RoboSpatial.tsv (Linux/Mac) or certutil -hashfile
        # RoboSpatial.tsv MD5 (Windows)
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self, dataset):
        """
        Load RoboSpatial dataset from TSV file via URL.
        """
        dataset = "RoboSpatial"
        url = self.DATASET_URL[dataset]
        md5 = self.DATASET_MD5.get(dataset, None)
        return self.prepare_tsv(url, md5)

    def point_in_polygon(self, x, y, poly):
        """
        Check if the point (x, y) lies within the polygon defined by a list of (x, y) tuples.
        Uses the ray-casting algorithm.
        """
        num = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, num + 1):
            p2x, p2y = poly[i % num]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def evaluate_answer(self, ground_truth, generated_answer):
        """
        Evaluates if the generated answer is correct based on the ground truth.
        Returns a tuple of (is_correct, is_binary_answer, parsed_answer, is_parsable).
        """
        gen_answer = generated_answer.strip().lower()
        gt_lower = ground_truth.strip().lower()

        # Check if this is a binary yes/no question
        if gt_lower in ["yes", "no"]:
            is_binary = True
            is_gt_yes = (gt_lower == "yes")
            # Binary answers are always considered parsable if they contain
            # text
            is_parsable = len(gen_answer) > 0
            if is_gt_yes:
                correct = gen_answer.startswith("yes")
            else:
                correct = gen_answer.startswith("no")
            return correct, is_binary, gen_answer, is_parsable
        else:
            # Numeric evaluation: ground_truth is a list of points defining a
            # polygon
            is_binary = False
            parsed_answer = None
            is_parsable = False  # Default to not parsable until we successfully parse

            try:
                gt_polygon = ast.literal_eval(ground_truth)
                if not isinstance(gt_polygon, list) or len(gt_polygon) < 3:
                    return False, is_binary, parsed_answer, is_parsable

                # Extract the first coordinate pair using regex
                # Look for patterns like (0.1,0.2) or (0.1, 0.2) or [0.1, 0.2]
                # or [0.1,0.2]

                # Try to match tuple format (x,y) or (x, y)
                tuple_match = re.search(
                    r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)',
                    generated_answer)
                if tuple_match:
                    try:
                        x = float(tuple_match.group(1))
                        y = float(tuple_match.group(2))
                        parsed_answer = (x, y)
                        is_parsable = True
                        correct = self.point_in_polygon(x, y, gt_polygon)
                        return correct, is_binary, parsed_answer, is_parsable
                    except (ValueError, TypeError):
                        pass

                # Try to match list format [x,y] or [x, y]
                list_match = re.search(
                    r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]',
                    generated_answer)
                if list_match:
                    try:
                        x = float(list_match.group(1))
                        y = float(list_match.group(2))
                        parsed_answer = (x, y)
                        is_parsable = True
                        correct = self.point_in_polygon(x, y, gt_polygon)
                        return correct, is_binary, parsed_answer, is_parsable
                    except (ValueError, TypeError):
                        pass

                # Fall back to parsing the full list
                try:
                    # Extract the first list from generated_answer
                    match = re.search(
                        r'\[(.*?)\]', generated_answer, re.DOTALL)
                    if match is None:
                        return False, is_binary, parsed_answer, is_parsable

                    list_content = match.group(1)
                    list_content = re.sub(r',(\S)', r', \1', list_content)
                    list_content = list_content.strip()
                    if list_content.endswith(','):
                        list_content = list_content[:-1]

                    list_str = '[' + list_content + ']'

                    try:
                        gen_val = ast.literal_eval(list_str)
                    except (SyntaxError, ValueError):
                        # If direct parsing fails, try to extract just the
                        # first tuple
                        tuple_match = re.search(
                            r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', list_content)
                        if tuple_match:
                            x = float(tuple_match.group(1))
                            y = float(tuple_match.group(2))
                            parsed_answer = (x, y)
                            is_parsable = True
                            correct = self.point_in_polygon(x, y, gt_polygon)
                            return correct, is_binary, parsed_answer, is_parsable
                        else:
                            return False, is_binary, parsed_answer, is_parsable

                    # Handle different formats for points
                    if isinstance(gen_val, list):
                        if len(gen_val) == 0:
                            return False, is_binary, parsed_answer, is_parsable

                        # Case 1: The list itself is a point coordinates [x, y]
                        if len(gen_val) == 2 and all(
                                isinstance(v, (int, float)) for v in gen_val):
                            gen_point = tuple(gen_val)
                        # Case 2: The list contains points [(x, y), ...]
                        elif isinstance(gen_val[0], tuple):
                            gen_point = gen_val[0]
                        # Case 3: The list contains coordinate pairs as lists
                        # [[x, y], ...]
                        elif isinstance(gen_val[0], list) and len(gen_val[0]) == 2:
                            gen_point = tuple(gen_val[0])
                        else:
                            return False, is_binary, parsed_answer, is_parsable
                    elif isinstance(gen_val, tuple):
                        gen_point = gen_val
                    else:
                        return False, is_binary, parsed_answer, is_parsable

                    if not (
                        isinstance(
                            gen_point,
                            tuple) and len(gen_point) == 2):
                        return False, is_binary, parsed_answer, is_parsable

                    x, y = float(gen_point[0]), float(gen_point[1])
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = self.point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except Exception:
                    # If all parsing attempts fail, return False
                    return False, is_binary, parsed_answer, is_parsable

            except Exception as e:
                print(f"Error evaluating answer: {e}")
                return False, is_binary, parsed_answer, is_parsable

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate the model predictions against ground truth.
        """
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        all_results = {
            "correct": 0,
            "total": 0,
            "format_error": 0,
            "illformed_responses": 0,
            "context": 0,
            "compatibility": 0,
            "configuration": 0,
            "context_correct": 0,
            "compatibility_correct": 0,
            "configuration_correct": 0,
            "results": []
        }

        for i in tqdm(range(len(lines)), desc="Evaluating RoboSpatial"):
            line = lines[i]
            index = int(line["index"])

            ground_truth = str(line["answer"])
            category = line.get("category", "unknown")

            # Parse the prediction
            if self.ROBUST:
                pred = line['prediction']
            else:
                # Try to extract structured answer if format is expected
                pred = line['prediction']

            # Evaluate the answer
            correct, is_binary, parsed_answer, is_parsable = self.evaluate_answer(
                ground_truth, pred)

            # Count illformed responses
            if not is_parsable:
                all_results["illformed_responses"] += 1
                all_results["format_error"] += 1

            all_results["total"] += 1
            if correct:
                all_results["correct"] += 1

            # Category-specific counting
            if category in all_results:
                all_results[category] += 1
                if correct:
                    all_results[f"{category}_correct"] += 1

            # Store detailed results
            result_entry = {
                "index": index,
                "question": line.get(
                    "question",
                    ""),
                "expected_answer": ground_truth,
                "generated_answer": pred,
                "parsed_answer": str(parsed_answer) if parsed_answer is not None else None,
                "correct": correct,
                "is_parsable": is_parsable,
                "category": category,
            }
            all_results["results"].append(result_entry)

        # Calculate overall accuracy
        all_results["score"] = all_results["correct"] / \
            all_results["total"] if all_results["total"] > 0 else 0.0

        # Add category-wise scores
        for category in ["context", "compatibility", "configuration"]:
            if all_results[category] > 0:
                all_results[f"{category}_score"] = (
                    all_results[f"{category}_correct"] / all_results[category]
                )
            else:
                all_results[f"{category}_score"] = 0.0

        # Save detailed results
        score_pth = eval_file.replace(".xlsx", "_score.json")
        dump(all_results, score_pth)

        return all_results

    def build_prompt(self, line):
        """
        Build the prompt for RoboSpatial questions.
        Different prompt styles for different categories.
        """
        msgs = super().build_prompt(line)

        return msgs
