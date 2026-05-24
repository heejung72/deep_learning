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
