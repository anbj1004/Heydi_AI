import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ---------------------------------------------------------
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 리스트 형태의 태그를 콤마로 구분된 문자열로 변환 (모델 학습용)
    # 예: ["급식", "학교생활"] -> "급식, 학교생활"
    df['target_text'] = df['tags'].apply(lambda x: ', '.join(x))
    return df

# 실제 사용 시 아래 주석 해제 후 json 파일 로드
df = load_data('model/dataset/diary_tag_dataset.json') 
# df = pd.DataFrame(sample_data)
df['target_text'] = df['tags'].apply(lambda x: ', '.join(x))

# 학습용과 검증용으로 분리 (데이터가 적으므로 test_size를 작게 잡음)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# ---------------------------------------------------------
# 2. 토크나이저 및 모델 불러오기 (KoBART)
# ---------------------------------------------------------
model_name = "gogamza/kobart-base-v1"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# ---------------------------------------------------------
# 3. 데이터셋 클래스 정의 (PyTorch Dataset)
# ---------------------------------------------------------
class DiaryTagDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=128, max_target_len=32):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = row['content']
        target_text = row['target_text']

        # 1. 입력 데이터 토큰화
        model_inputs = self.tokenizer(
            input_text, 
            max_length=self.max_input_len, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )

        # 2. 정답(태그) 데이터 토큰화 (수정)
        # as_target_tokenizer 대신 text_target 인자 사용
        labels = self.tokenizer(
            text_target=target_text,
            max_length=self.max_target_len, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )

        # Labels에서 Padding 토큰은 Loss 계산 시 제외하기 위해 -100으로 변경
        labels_ids = labels["input_ids"].squeeze()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": labels_ids
        }

train_dataset = DiaryTagDataset(train_df, tokenizer)
val_dataset = DiaryTagDataset(val_df, tokenizer)

# ---------------------------------------------------------
# 4. 학습 설정 (Training Arguments)
# ---------------------------------------------------------
# 데이터가 적으므로 epoch를 조금 넉넉히 주되, batch_size는 작게 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",           # 결과 저장 경로
    num_train_epochs=10,              # 학습 횟수 (데이터가 적어서 10~20회 추천)
    per_device_train_batch_size=4,    # 배치 크기
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,      # 가장 성능 좋은 모델 저장
    metric_for_best_model="eval_loss",
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,               # 용량 절약을 위해 최근 모델 2개만 저장
    predict_with_generate=True,       # 평가 시 생성 모드 활성화
    fp16=torch.cuda.is_available(),   # GPU 사용 시 속도 향상 (Mixed Precision)
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ---------------------------------------------------------
# 5. 트레이너 정의 및 학습 시작
# ---------------------------------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print(">>> 학습을 시작합니다...")
trainer.train()

# 학습된 모델 저장
model.save_pretrained("./final_diary_tag_model")
tokenizer.save_pretrained("./final_diary_tag_model")
print(">>> 학습 완료 및 모델 저장됨.")

# ---------------------------------------------------------
# 6. 추론 (Inference) 테스트
# ---------------------------------------------------------
def generate_tags_cleaned(input_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True, 
        padding="max_length"
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 1. 모델 생성 (파라미터 튜닝)
    summary_text_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=64,
        num_beams=5,             # 빔 서치 개수 증가
        repetition_penalty=2.0,  # 반복 페널티를 더 강하게 (1.2 -> 2.0)
        no_repeat_ngram_size=2,  # 2단어 이상 겹치는 구문 생성 금지 (핵심)
        early_stopping=True
    )

    output = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    
    # 2. 후처리 (Python String Manipulation)
    # 콤마(,) 기준으로 자르고 앞뒤 공백 제거
    tags = [tag.strip() for tag in output.split(',')]
    
    # 빈 문자열 제거 및 중복 제거 (순서 유지)
    seen = set()
    cleaned_tags = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            cleaned_tags.append(tag)
            
    # 최대 2개까지만 반환
    return cleaned_tags[:2]

# 테스트
test_diary = "오늘 여자친구랑 헤어졌다. 아무것도 하기 싫고 밥맛도 없다. 그냥 침대에 누워서 하루 종일 천장만 바라봤다. 눈물도 안 나온다."
final_tags = generate_tags_cleaned(test_diary)

print(f"입력: {test_diary}")
print(f"결과: {final_tags}") 
# 예상 결과: ['이별', '무기력'] 또는 ['이별', '슬픔'] 등