# Deep Learning Study

## 1주차

📺 [파이토치 딥러닝 강의](https://www.youtube.com/watch?v=k60oT_8lyFw&list=PL7ZVZgsnLwEEIC4-KQIchiPda_EjxX61r)

### 공부 내용

- **Autograd (자동 미분)**
  - `requires_grad=True`로 텐서의 기울기 추적 시작
  - `backward()`로 자동으로 모든 기울기 계산
  - `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` 순서로 학습

- **zero_grad() 왜 필요한가**
  - PyTorch는 `backward()` 호출 시 기존 `.grad`에 누적(`+=`)하는 방식으로 동작
  - `step()`으로 이미 반영한 기울기가 다음 배치에도 남아있어 학습이 망가짐
  - 매 배치 시작 전 `zero_grad()`로 초기화 필수

---

## 2주차

📺 NLP 토큰화 & 임베딩 실습

### 공부 내용

- **Sentence-Transformers 임베딩**
  - `paraphrase-multilingual-MiniLM-L12-v2` 모델로 한국어 텍스트 임베딩 생성
  - 임베딩 히트맵으로 각 텍스트의 벡터 패턴 시각화
  - 코사인 유사도(Cosine Similarity)로 텍스트 간 의미적 거리 측정

- **TF-IDF 벡터화**
  - `TfidfVectorizer`로 단어 빈도 기반 벡터 생성
  - 임베딩과 TF-IDF 방식의 차이 비교

---

## 3주차

📺 [Attention/Transformer 시각화로 설명](https://www.youtube.com/watch?v=6s69XY025MU) | [Transformer: Attention Is All You Need - 나동빈](https://www.youtube.com/watch?v=AA621UofTUA)

### 공부 내용

- **Transformer 아키텍처**
  - RNN/LSTM의 순차 처리 한계(long-range dependency, vanishing gradient) 극복
  - Encoder 6개 + Decoder 6개 블록 구조
  - Q(Query), K(Key), V(Value) 행렬로 Scaled Dot-Product Attention 수행
  - Multi-Head Attention으로 문법·의미·위치 등 다양한 관계 병렬 포착

- **Positional Encoding**
  - Transformer는 순서 개념이 없어 sin/cos 함수로 위치 정보 주입

- **BERT**
  - Transformer의 Encoder만 사용하는 양방향 사전학습 언어 모델
  - MLM(Masked Language Model) + NSP(Next Sentence Prediction)로 사전학습
  - Fine-tuning으로 분류, QA 등 다운스트림 태스크에 적용

- **번역 실습 (독일어 → 영어)**
  - Multi30k 데이터셋, spaCy 토크나이저 사용
  - 10 에포크 학습 → BLEU Score 35.38 달성
