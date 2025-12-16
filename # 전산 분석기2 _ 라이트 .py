# 라이트 아기동진이
# ============================================================
# 0. 라이브러리 불러오기
# ============================================================
import urllib.request
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW


# ============================================================
# 1. NSMC 데이터 다운로드 (ratings_train.txt, ratings_test.txt)
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
# 2. 데이터 로드 및 '라이트 버전' 샘플링
# ============================================================
print("\n데이터 로드 중...")
train_df = pd.read_table(train_path)   # id, document, label
test_df  = pd.read_table(test_path)

# 결측치 제거
train_df = train_df.dropna()
test_df  = test_df.dropna()

# 🔹 가벼운 버전: train 5,000개 / test 5,000개만 사용
train_small = train_df.sample(5000, random_state=42)
test_small  = test_df.sample(5000,  random_state=42)

print(f"train_small 크기: {len(train_small)}")
print(f"test_small  크기: {len(test_small)}")

X_train, X_valid, y_train, y_valid = train_test_split(
    train_small["document"],
    train_small["label"],
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. BERT(KcBERT) 토크나이저 / 모델 / 디바이스 설정
# ============================================================
MODEL_NAME = "beomi/kcbert-base"   # 댓글/리뷰 도메인에 잘 맞는 KcBERT

# 🔹 가벼운 하이퍼파라미터 세팅 (CPU 친화)
MAX_LEN   = 64
BATCH_SIZE = 16
EPOCHS     = 1

print("\n토크나이저와 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2   # 0: 부정, 1: 긍정
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"사용 디바이스: {device}")


# ============================================================
# 4. Dataset / DataLoader 정의
# ============================================================
class NSMCDataset(Dataset):
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
test_dataset  = NSMCDataset(test_small["document"], test_small["label"], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print("\nDataLoader 준비 완료.")
print(f"  train 배치 수: {len(train_loader)}")
print(f"  valid 배치 수: {len(valid_loader)}")
print(f"  test  배치 수: {len(test_loader)}")


# ============================================================
# 5. 학습 / 예측 함수 정의
# ============================================================
def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    print(f"[train_one_epoch 시작] 미니배치 개수: {len(data_loader)}")

    for step, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # 진행 상황 로그 (가벼운 버전이라 step 간격 줄임)
        if (step + 1) % 50 == 0:
            print(f"  [step {step+1}/{len(data_loader)}] loss = {loss.item():.4f}")

    avg_loss = total_loss / max(1, step + 1)
    print(f"[train_one_epoch 종료] 평균 loss = {avg_loss:.4f}")
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
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1)

            preds.extend(batch_preds.cpu().numpy().tolist())
            labels_list.extend(labels.numpy().tolist())

    return preds, labels_list


# ============================================================
# 6. Optimizer / Scheduler 설정
# ============================================================
optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# ============================================================
# 7. 학습 루프 (라이트 버전: EPOCHS=1)
# ============================================================
for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)

    val_preds, val_labels = predict(model, valid_loader, device)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")


# ============================================================
# 8. 테스트 셋 평가 (라이트: test_small 기준)
# ============================================================
test_preds, test_labels = predict(model, test_loader, device)
test_acc = accuracy_score(test_labels, test_preds)
print("\n===== Test 결과 (BERT 라이트 버전) =====")
print("Test Accuracy:", test_acc)
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, digits=4))


# ============================================================
# 9. 한 개 리뷰 예측 함수 (확률까지 반환)
# ============================================================
def predict_single_review(text, model, tokenizer, max_len, device):
    model.eval()

    with torch.no_grad():
        encoding = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        encoding = {k: v.to(device) for k, v in encoding.items()}

        outputs = model(**encoding)
        logits = outputs.logits          # [1, 2]
        probs_tensor = torch.softmax(logits, dim=1)[0]

        probs = probs_tensor.cpu().tolist()          # [p_neg, p_pos]
        pred_label = int(torch.argmax(probs_tensor).item())

    return pred_label, probs


# ============================================================
# 10. 긍정 확률 → 0~5점 (0.5 단위) 별점으로 변환
# ============================================================
def prob_to_star_rating(p_pos: float) -> float:
    """
    p_pos: 긍정 확률 (0.0 ~ 1.0)
    반환: 0.0 ~ 5.0 사이 0.5 단위 별점
    """
    raw_score = p_pos * 5.0                 # 0~5 실수값
    half_step = round(raw_score * 2) / 2.0  # 0.5 단위로 반올림
    return max(0.0, min(5.0, half_step))    # 0~5 범위로 클램프


# ============================================================
# 11. 실시간 리뷰 입력 → 0~5점 별점 예측기
# ============================================================
print("\n🔹 실시간 리뷰 별점 예측기 (0~5점, 0.5 단위)")
print("   리뷰를 입력하면 BERT가 별점을 매겨줍니다. (종료: q)")

while True:
    text = input("\n리뷰를 입력하세요 (종료하려면 q): ").strip()
    if text.lower() == "q":
        print("종료합니다. 오늘도 고생 많았어 두목 💚")
        break

    if not text:
        print("공백 말고 내용을 좀 써줘요 두목 🥺")
        continue

    pred_label, probs = predict_single_review(text, model, tokenizer, MAX_LEN, device)
    p_neg, p_pos = probs[0], probs[1]

    rating = prob_to_star_rating(p_pos)

    print(f"\n[모델 예측]")
    print(f"  부정 확률 = {p_neg:.3f}, 긍정 확률 = {p_pos:.3f}")
    print(f"  → 예측 별점 = ⭐ {rating:.1f} / 5.0")

    if pred_label == 1:
        print("  (전체 판단: 긍정 쪽에 가깝습니다.)")
    else:
        print("  (전체 판단: 부정 쪽에 가깝습니다.)")
