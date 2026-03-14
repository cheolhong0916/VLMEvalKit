import pandas as pd
import json
import os
import sys

def evaluate_erqa_xlsx(xlsx_path):
    """Evaluate ERQA benchmark from xlsx file with CoT support and specific print format"""
    print(f"Reading file from: {xlsx_path}")
    
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"Error reading excel file: {e}")
        return None
    
    # 1. Prediction 컬럼 찾기
    pred_col = None
    for col in df.columns:
        if 'prediction' in col.lower() or col.lower() == 'pred':
            pred_col = col
            break
    
    if pred_col is None:
        print("ERROR: Could not find prediction column in xlsx file")
        return None
    
    # 2. Answer 컬럼 찾기
    answer_col = None
    for col in df.columns:
        if col.lower() in ['answer', 'gt', 'ground_truth', 'label']:
            answer_col = col
            break
    
    if answer_col is None:
        print("ERROR: Could not find answer column in xlsx file")
        return None

    # 3. Image Type, Question Type 컬럼 찾기
    image_type_col = None
    for col in df.columns:
        if 'image' in col.lower() and 'type' in col.lower():
            image_type_col = col
            break
            
    question_type_col = None
    for col in df.columns:
        if 'question' in col.lower() and 'type' in col.lower():
            question_type_col = col
            break

    # 4. 통계 집계 초기화
    results = {}
    total_correct = 0
    total_count = 0
    
    # 이미지 타입별 통계 (Single, Multi)
    img_stats = {
        'Single_Image': {'correct': 0, 'total': 0}, 
        'Multi_Image': {'correct': 0, 'total': 0}
    }
    
    # 질문 타입별 통계 (순서 중요)
    q_type_order = [
        'Trajectory Reasoning', 'Action Reasoning', 'Pointing', 
        'State Estimation', 'Spatial Reasoning', 'Multi-view Reasoning', 
        'Task Reasoning', 'Other'
    ]
    q_stats = {qt: {'correct': 0, 'total': 0} for qt in q_type_order}

    # 5. Row별 평가 루프
    for idx, row in df.iterrows():
        gt = row[answer_col]
        pred = row[pred_col]
        
        if pd.isna(gt):
            continue
            
        total_count += 1
        
        # Ground Truth 정규화
        gt_str = str(gt).replace('.', '').strip().upper()
        
        # Prediction 정규화 (CoT 처리)
        pred_str = str(pred)
        if "</think>" in pred_str:
            # </think> 태그 뒤의 내용만 가져옴
            pred_str = pred_str.split("</think>")[-1]
            
        clean_pred = pred_str.replace('.', '').strip().upper()
        
        # "Option A" 또는 "A ..." 등의 형태 처리
        if clean_pred.startswith("OPTION ") and len(clean_pred) > 7:
             clean_pred = clean_pred.split(" ")[1]
        elif len(clean_pred) > 1 and clean_pred[0] in ['A', 'B', 'C', 'D', 'E'] and clean_pred[1] == ' ':
             clean_pred = clean_pred[0]
             
        # 정답 여부 판단
        is_correct = (clean_pred == gt_str)
        if is_correct:
            total_correct += 1
            
        # Image Type 별 집계
        if image_type_col and not pd.isna(row[image_type_col]):
            it = str(row[image_type_col]).strip()
            # 데이터 내 타입명이 Single_Image, Multi_Image와 정확히 일치한다고 가정
            if it in img_stats:
                img_stats[it]['total'] += 1
                if is_correct:
                    img_stats[it]['correct'] += 1
        
        # Question Type 별 집계
        if question_type_col and not pd.isna(row[question_type_col]):
            qt = str(row[question_type_col]).strip()
            if qt in q_stats:
                q_stats[qt]['total'] += 1
                if is_correct:
                    q_stats[qt]['correct'] += 1
    
    # 6. 결과 계산 및 저장
    results['Correct'] = total_correct
    results['Total'] = total_count
    results['Accuracy'] = (total_correct / total_count) if total_count > 0 else 0.0
    
    # Image Type 결과
    for it in ['Single_Image', 'Multi_Image']:
        s = img_stats[it]
        results[f'{it}_Correct'] = s['correct']
        results[f'{it}_Total'] = s['total']
        results[f'{it}_Accuracy'] = (s['correct'] / s['total']) if s['total'] > 0 else 0.0
        
    # Question Type 결과
    for qt in q_type_order:
        s = q_stats[qt]
        results[f'{qt}_Correct'] = s['correct']
        results[f'{qt}_Total'] = s['total']
        results[f'{qt}_Accuracy'] = (s['correct'] / s['total']) if s['total'] > 0 else 0.0

    # 7. 출력 순서 정의 (요청하신 포맷 순서)
    output_order = [
        'Correct', 'Total', 'Accuracy',
        'Single_Image_Accuracy', 'Multi_Image_Accuracy'
    ]
    # QType Accuracies 추가
    output_order.extend([f'{qt}_Accuracy' for qt in q_type_order])
    
    # Image Counts 추가
    output_order.extend([
        'Single_Image_Correct', 'Single_Image_Total',
        'Multi_Image_Correct', 'Multi_Image_Total'
    ])
    
    # QType Counts (Correct, Total 쌍) 추가
    for qt in q_type_order:
        output_order.extend([f'{qt}_Correct', f'{qt}_Total'])

    # 8. 최종 출력
    print("\nKeys in the JSON file:")
    for k in output_order:
        print(k)
        
    print("\nValues in the JSON file:")
    for k in output_order:
        val = results.get(k, 0)
        if 'Accuracy' in k or k == 'Accuracy':
            print(f"{float(val)}")  # Accuracy는 float 출력 (예: 0.0)
        else:
            print(f"{int(val)}")    # Count는 int 출력

    return results

if __name__ == "__main__":
    # 경로 지정 (필요에 따라 수정)
    xlsx_path = "/data/shared/Qwen/VLMEvalKit/outputs/molmo-7B-O-0924-stage2_hard_cot_in_400k/molmo-7B-O-0924-stage2_hard_cot_in_400k_ERQA.xlsx"
    
    if os.path.exists(xlsx_path):
        evaluate_erqa_xlsx(xlsx_path)
    else:
        print(f"Error: File not found at {xlsx_path}")