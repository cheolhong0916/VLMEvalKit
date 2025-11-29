"""
EmbSpatial-Bench Dataset for VLMEvalKit

A benchmark for evaluating embodied spatial understanding of Vision-Language Models.
- 3,640 QA pairs across 294 object categories
- 6 spatial relationships from egocentric perspective: far, left, right, above, below, behind
- Multiple-choice questions (4 options)

Reference:
    - Paper: https://arxiv.org/abs/2410.20062
    - Dataset: https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench
"""

from .image_mcq import ImageMCQDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings


class EmbSpatialBench(ImageMCQDataset):
    """
    EmbSpatial-Bench: Embodied Spatial Understanding Benchmark
    
    Evaluates VLM understanding of spatial relationships in embodied environments.
    """
    
    TYPE = 'MCQ'
    
    DATASET_URL = {
        'EmbSpatialBench': 'https://huggingface.co/ch-min/EmbSpatial-Bench-tsv/resolve/main/EmbSpatial-Bench.tsv'
    }
    
    DATASET_MD5 = {
        'EmbSpatialBench': 'fc6c2ec0d7e9d454602fba7fb933f1fa'
    }
    
    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate with overall accuracy + per-relation breakdown
        
        Returns:
            pd.DataFrame with overall accuracy and per-relation accuracies
        """
        from .utils.multiple_choice import report_acc, mcq_vanilla_eval
        
        # Setup evaluation
        nproc = judge_kwargs.pop('nproc', 4)
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map.get(model, model)
        
        # Build judge if needed
        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None
        
        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')
        
        # Load and preprocess data
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        
        # Normalize column names (lowercase except A-D)
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)
        
        # Validate data
        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )
        
        # Perform MCQ evaluation
        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        
        # Save results
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        
        # Calculate overall accuracy
        acc = report_acc(data)
        
        # Calculate per-relation accuracy breakdown
        acc_by_relation = self.report_acc_by_relation(data)
        
        # Save results
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)
        
        score_file_relation = eval_file.replace(f'.{suffix}', '_acc_by_relation.csv')
        dump(acc_by_relation, score_file_relation)
        
        print("\n=== EmbSpatial-Bench Evaluation Results ===")
        print(f"Overall Accuracy: {acc['Overall'].iloc[0]:.4f}")
        print("\nPer-Relation Accuracy:")
        print(acc_by_relation.to_string())
        
        return acc
    
    def report_acc_by_relation(self, df):
        """
        Calculate accuracy for each spatial relation
        
        Args:
            df: DataFrame with 'category' (relation) and 'hit' (correctness) columns
            
        Returns:
            pd.DataFrame with per-relation accuracies
        """
        res = defaultdict(list)
        
        # Check for split column
        if 'split' in df:
            splits = list(set(df['split']))
            res['split'] = splits
        else:
            df['split'] = ['none'] * len(df)
            res['split'] = ['none']
        
        # Overall accuracy
        res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]
        
        # Per-relation accuracy
        if 'category' in df:
            relations = list(set(df['category']))
            relations.sort()
            
            for rel in relations:
                sub_df = df[df['category'] == rel]
                res[rel] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
        
        # Per-data_source accuracy (optional, if column exists)
        if 'data_source' in df:
            sources = list(set(df['data_source']))
            sources = ['None' if isinstance(s, float) and pd.isna(s) else s for s in sources]
            sources.sort()
            
            for src in sources:
                sub_df = df[df['data_source'] == src]
                res[f'source_{src}'] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]
        
        return pd.DataFrame(res)


# For backwards compatibility or alternative naming
class EmbSpatialBenchDataset(EmbSpatialBench):
    """Alias for EmbSpatialBench"""
    pass