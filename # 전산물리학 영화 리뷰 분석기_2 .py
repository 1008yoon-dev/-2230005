# 
# 이번엔 KObert,kcbert kc ko electra를 사용해보겟음
# ============================================================
# 0. 라이브러리 불러오기
# ============================================================
import urllib.request           # GitHub에서 데이터 파일을 다운로드하기 위한 라이브러리
import os                       # 파일 존재 여부 확인용
import pandas as pd             # 표 형태 데이터(DataFrame) 다루기
from sklearn.model_selection import train_test_split   # train/valid 분할
from sklearn.metrics import accuracy_score, classification_report  # 평가 지표

import torch                    # [*1]
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup   # [*3-1] 스케줄러만 그대로 transformers에서 사용
)

from torch.optim import AdamW      
    # [*3-2] AdamW는 이제 torch.optim에서 가져옴

# ============================================================
# 1. NSMC 데이터 다운로드 (ratings_train.txt, ratings_test.txt)
#    - 이미 파일이 있으면 다시 안 받도록 체크
# ============================================================
train_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
test_url  = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
train_path = "ratings_train.txt"
test_path  = "ratings_test.txt"

if not os.path.exists(train_path):
    print("NSMC train 데이터 다운로드 중...")
    urllib.request.urlretrieve(train_url, filename=train_path)
    print("다운로드 완료!")

if not os.path.exists(test_path):
    print("NSMC test 데이터 다운로드 중...")
    urllib.request.urlretrieve(test_url, filename=test_path)
    print("다운로드 완료!")

# ============================================================
# 2. 데이터 로드 및 기본 전처리
# ============================================================
train_df = pd.read_table(train_path)   # id, document, label
test_df  = pd.read_table(test_path)

# 결측치 제거
train_df = train_df.dropna()
test_df  = test_df.dropna()

# (기존에 쓰던 한글/특수문자 정리 함수가 있다면 여기서 그대로 사용 가능)
# 예: train_df['document'] = train_df['document'].apply(clean_korean_text)

# train/valid 분할 (test_df는 공식 테스트셋 그대로 사용)
X_train, X_valid, y_train, y_valid = train_test_split(
    train_df['document'],
    train_df['label'],
    test_size=0.2,
    random_state=42
)

# ============================================================
# 3. BERT 토크나이저/모델 준비
# ============================================================
MODEL_NAME = "beomi/kcbert-base"   # [*4] 한국어 댓글/대화체에 잘 맞는 KcBERT

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # [*5]
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2                                         # [*6]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)                                         # [*7]

MAX_LEN = 128                                            # [*8]
BATCH_SIZE = 32
EPOCHS = 1                                               # [*9]

# ============================================================
# 4. Dataset / DataLoader 정의 (문자열 → 토큰 ID 배치)
# ============================================================
class NSMCDataset(Dataset):                              # [*10]
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = NSMCDataset(X_train, y_train, tokenizer, MAX_LEN)
valid_dataset = NSMCDataset(X_valid, y_valid, tokenizer, MAX_LEN)
test_dataset  = NSMCDataset(test_df["document"], test_df["label"], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 5. 학습/평가 함수 정의
# ============================================================
def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for  batch in data_loader:
        # 배치를 GPU/CPU로 옮기기
        batch = {k: v.to(device) for k, v in batch.items()}   # [*11]

        outputs = model(**batch)   # forward pass, 내부에서 cross-entropy loss 계산
        loss = outputs.loss

        loss.backward()            # backpropagation
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def predict(model, data_loader, device):
    model.eval()
    preds = []
    labels_list = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits          # [*12]
            batch_preds = torch.argmax(logits, dim=1)

            preds.extend(batch_preds.cpu().numpy().tolist())
            labels_list.extend(labels.numpy().tolist())

    return preds, labels_list

# ============================================================
# 6. Optimizer / Scheduler 설정 및 학습 루프
# ============================================================
optimizer = AdamW(model.parameters(), lr=2e-5)          # [*13]

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(            # [*14]
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Train loss: {train_loss:.4f}")

    val_preds, val_labels = predict(model, valid_loader, device)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")

# ============================================================
# 7. 테스트 셋 평가
# ============================================================
test_preds, test_labels = predict(model, test_loader, device)
test_acc = accuracy_score(test_labels, test_preds)
print("\n===== Test 결과 (BERT) =====")
print("Test Accuracy:", test_acc)
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, digits=4))
