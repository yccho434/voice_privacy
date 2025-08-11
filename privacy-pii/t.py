from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 존재하는 공개 모델로 교체!
model_id = "KPF/KPF-bert-ner"   # 또는 "taeminlee/gliner_ko"

tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForTokenClassification.from_pretrained(model_id)
nlp = pipeline("token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple")

text = "김민수님이 2025-08-18에 서울특별시 마포구 성산로 42길 15로 이사했습니다. 이메일은 jiyeongpark@example.com 입니다."
print(nlp(text))
