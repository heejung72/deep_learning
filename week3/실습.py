# 실습: Transformer 기반 독일어 → 영어 번역
# 영상 2 ([딥러닝 기계 번역] Transformer: Attention Is All You Need) 실습 재현
# 데이터셋: Multi30k (독일어-영어 병렬 코퍼스)
#
# 실행 전 설치:
#   pip install transformers sentencepiece

from transformers import MarianMTModel, MarianTokenizer

# Helsinki-NLP의 사전학습된 독일어 → 영어 Transformer 모델 로드
model_name = "Helsinki-NLP/opus-mt-de-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate_de_to_en(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


# Multi30k 스타일 예시 문장
test_sentences = [
    "Eine Frau mit einer großen Geldbörse.",          # 영상 예시 문장
    "Ein Mann sitzt auf einer Bank im Park.",
    "Zwei Kinder spielen im Schnee.",
    "Der Hund läuft durch den Wald.",
    "Ein Wissenschaftler erklärt die Quantenmechanik.",
]

print("=" * 60)
print("독일어 → 영어 번역 (Transformer / Attention Is All You Need)")
print("=" * 60)

for de in test_sentences:
    en = translate_de_to_en(de)
    print(f"\n[DE] {de}")
    print(f"[EN] {en}")

# 직접 입력 번역
print("\n" + "=" * 60)
print("직접 번역해보기 (종료: 'q' 입력)")
print("=" * 60)

while True:
    user_input = input("\n[DE] 독일어 문장 입력: ").strip()
    if user_input.lower() == "q":
        break
    if user_input:
        result = translate_de_to_en(user_input)
        print(f"[EN] {result}")
