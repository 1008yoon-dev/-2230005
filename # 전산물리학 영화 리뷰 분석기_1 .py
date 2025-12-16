# 진짜 기말 과제


# ============================================================
# 0. 라이브러리 불러오기
# ============================================================
import urllib.request           # GitHub에서 데이터 파일을 다운로드하기 위한 라이브러리
import os                       # 파일 존재 여부 확인용
import pandas as pd             # 표 형태 데이터(DataFrame) 다루기
from sklearn.model_selection import train_test_split   # train/test 분할
from sklearn.feature_extraction.text import TfidfVectorizer  # 텍스트 → TF-IDF 벡터화
from sklearn.linear_model import LogisticRegression   # 로지스틱 회귀(이진/다중 분류)
from sklearn.metrics import accuracy_score, classification_report  # 평가 지표

# ============================================================
# 1. NSMC 데이터 다운로드 (ratings_train.txt)
#    - 이미 파일이 있으면 다시 안 받도록 체크
# ============================================================
train_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
train_path = "ratings_train.txt"   # 저장할 파일 이름

if not os.path.exists(train_path):
    print("NSMC train 데이터 다운로드 중...")
    urllib.request.urlretrieve(train_url, filename=train_path)
    print("다운로드 완료:", train_path)
else:
    print("이미 존재하는 파일 사용:", train_path)

# ============================================================
# 2. 데이터 로드 및 기본 전처리
#    - ratings_train.txt는 탭(\t)으로 구분된 txt 파일
#    - 컬럼: id, document(리뷰 텍스트), label(0/1)
# ============================================================
df = pd.read_csv(train_path, sep="\t")

print("원본 데이터 상위 5개:")
print(df.head())

# 결측치 제거 (document 또는 label이 비어있는 행은 삭제)
df = df.dropna(subset=["document", "label"])

# label을 정수형으로 캐스팅 (혹시 모를 타입 문제 방지)
df["label"] = df["label"].astype(int)

# ============================================================
# 3. 입력(X)과 타깃(y) 정의
#    - X_text: 리뷰 텍스트
#    - y: 0(부정) / 1(긍정)
# ============================================================
X_text = df["document"]
y = df["label"]

# ============================================================
# 4. Train:Test = 9:1 분할
#    - stratify=y 로 클래스 비율 유지
# ============================================================
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.1,      # 10%를 test로 사용 → 9:1 분할
    random_state=42,    # 재현성을 위한 시드 고정
    stratify=y          # 긍/부정 비율이 train/test에 비슷하게 가도록
)

print("Train 개수:", len(X_train_text))
print("Test 개수:", len(X_test_text))

# ============================================================
# 5. TF-IDF 벡터화
#    - 한국어도 기본적으로 공백 단위 토크나이즈라 어느 정도 동작함
#    - 고급 tokenizer를 써도 되지만, 예제에선 기본값으로 진행
# ============================================================


# ============================================================
# 6. 로지스틱 회귀 모델 정의 및 학습
#    - 이진 분류: label 0(부정), 1(긍정)
#    - penalty='l2' : Ridge 정규화 (2-노름) → 과적합 방지
# ============================================================
def train_model(C=1.8):
    tfidf = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        min_df=5,
    )
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    model = LogisticRegression(
        penalty='l2',
        C=C,
        solver='liblinear',
        max_iter=3000,
    )
    model.fit(X_train, y_train)
    return tfidf, model, X_test

print("모델 학습")
tfidf, model, X_test = train_model(C=1.8)   # 여기서 전역 변수로 받기(변수 범위 설정에 유의하기)
print("학습 완료")

# ============================================================
# 7. 테스트 데이터 평가
# ============================================================
y_pred = model.predict(X_test)         # 0/1 예측
y_proba = model.predict_proba(X_test)  # [P(부정), P(긍정)] 확률

# 정확도 출력
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# 클래스별 precision / recall / f1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ============================================================
# 8. 새 리뷰 한 줄을 받아서
#    - 0/1 긍/부정 판단
#    - 확률 기반 0~5 별점 매기기
# ============================================================
def analyze_review(text: str):
    """
    text : 리뷰 텍스트 (문자열)

    return:
        star_05 : 0 ~ 5 사이, 0.5 단위 별점
        sentiment : 'negative' / 'neutral' / 'positive'
        p_neg, p_pos : 부정/긍정 확률
    """
    # 1) 텍스트 → TF-IDF 벡터
    vec = tfidf.transform([text])

    # 2) 부정/긍정 확률 예측
    proba = model.predict_proba(vec)[0]
    p_neg = proba[0]
    p_pos = proba[1]

    # 3) 0~5 구간으로 선형 매핑
    raw_star = p_pos * 5.0   # p_pos ∈ [0,1] → raw_star ∈ [0,5]

    # 4) 0.5 단위로 반올림
    star_05 = round(raw_star * 2) / 2.0   # 예: 3.26 → 6.52 → round(6.52)=7 → 7/2=3.5

    # (옵션) 0점은 쓰고 싶지 않으면 아래 주석 풀기
    # star_05 = max(0.5, star_05)

    # 5) 별점으로 감성 텍스트 라벨 정하기 (기준은 취향대로 조정 가능)
    if star_05 >= 3.5:
        sentiment = "positive"
    elif star_05 <= 1.5:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return star_05, sentiment, p_neg, p_pos


# ============================================================
# 10. 터미널에서 직접 새 리뷰를 입력해서 분석해보기
#     - 위의 analyze_review() 함수를 그대로 사용
# ============================================================
if __name__ == "__main__":
    print("\n=== 영화 리뷰 감성/별점 분석기 ===")
    print("리뷰를 입력하면 [부정/중립/긍정]과 1~3점 별점을 알려줄게요.")
    print("종료하려면 q 또는 quit 를 입력하세요.\n")

    while True:
        text = input("리뷰 입력: ").strip()

        # 종료 조건
        if text.lower() in ("q", "quit", "exit"):
            print("종료할게요.")
            break

        # 분석 실행
        star, senti, p_neg, p_pos = analyze_review(text)

        print(f" → 감성: {senti}")
        print(f" → 별점 (1~3): {star}")
        print(f" → P(부정) = {p_neg:.3f}, P(긍정) = {p_pos:.3f}")
        print("-" * 50)
