import pandas as pd
import sys
import os

# ----------------------------------------------------------------------
# [설정] 로컬 파일 경로 (필요시 수정)
# ----------------------------------------------------------------------
# 1. 인터넷에서 직접 다운로드할 URL
TSV_URL = "https://huggingface.co/datasets/ch-min/ERQA-tsv/resolve/main/erqa.tsv"

# 2. (선택 사항) 파일을 수동으로 다운로드한 경우, 이 변수에 로컬 파일 경로를 지정하세요.
# 예: LOCAL_FILE_PATH = "/data/shared/Qwen/VLMEvalKit/erqa.tsv"
LOCAL_FILE_PATH = None
# ----------------------------------------------------------------------


def main():
    if LOCAL_FILE_PATH and os.path.exists(LOCAL_FILE_PATH):
        print(f"로컬 파일에서 ERQA TSV를 로드합니다...")
        file_to_load = LOCAL_FILE_PATH
    else:
        print(f"Hugging Face URL에서 ERQA TSV 파일을 로드합니다...")
        print(f"URL: {TSV_URL}")
        file_to_load = TSV_URL

    try:
        # URL 또는 로컬 파일에서 TSV 파일을 DataFrame으로 로드합니다.
        df = pd.read_csv(file_to_load, sep='\t')
        print(f"성공! 총 {len(df)}개의 행을 로드했습니다.")
    except Exception as e:
        print(f"TSV 파일 로드 실패: {e}", file=sys.stderr)
        if not LOCAL_FILE_PATH:
            print("인터넷 연결을 확인하거나, 파일을 수동으로 다운로드한 후 LOCAL_FILE_PATH 변수를 설정해주세요.", file=sys.stderr)
        sys.exit(1)

    # erqa.py에서 확인한 16개의 이미지 컬럼 이름 리스트
    image_columns = ['image'] + [f'image{i}' for i in range(2, 17)]

    # 0개의 이미지를 가진 행을 저장할 리스트
    rows_with_zero_images = []

    print("\n--- 각 행의 이미지 개수 스캔 시작 ---")

    # DataFrame의 모든 행을 순회합니다.
    for index, row in df.iterrows():
        current_image_count = 0
        
        # 16개 이미지 컬럼을 모두 확인합니다.
        for col_name in image_columns:
            # 1. 컬럼이 DataFrame에 존재하는지 확인
            if col_name in row:
                cell_value = row[col_name]
                
                # 2. erqa.py의 로직과 동일하게 체크:
                #    - pd.isna(cell_value) == False (NaN이 아님)
                #    - isinstance(cell_value, str) == True (문자열임)
                #    - cell_value.strip() != "" (빈 문자열이나 공백이 아님)
                if not pd.isna(cell_value) and isinstance(cell_value, str) and cell_value.strip():
                    current_image_count += 1
                    
        # 이 행의 이미지 개수가 0개인지 확인
        if current_image_count == 0:
            rows_with_zero_images.append({
                "dataframe_row": index,  # 0부터 시작하는 pandas 행 인덱스
                "index_column_value": row.get('index', 'N/A'), # TSV의 'index' 컬럼 값
                "question": str(row.get('question', 'N/A'))[:100] + "..." # 질문 (너무 길면 자름)
            })

    print("--- 스캔 완료 ---")

    # 최종 결과 보고
    if not rows_with_zero_images:
        print(f"\n[성공] 모든 {len(df)}개의 행이 최소 1개 이상의 이미지를 포함하고 있습니다.")
    else:
        print(f"\n[!!경고!!] 총 {len(df)}개의 행 중 {len(rows_with_zero_images)}개에서 이미지가 발견되지 않았습니다:")
        print("-" * 50)
        for item in rows_with_zero_images:
            print(f"  - DataFrame 행 번호: {item['dataframe_row']}")
            print(f"    'index' 컬럼 값: {item['index_column_value']}")
            print(f"    질문: {item['question']}")
            print("-" * 50)

if __name__ == "__main__":
    main()