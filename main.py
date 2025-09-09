# ============================
# 1⃣ CÀI ĐẶT THƯ VIỆN
# ============================
#!pip install transformers -q

# ============================
# 2⃣ IMPORT
# ============================
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ============================
# 3⃣ TẢI MODEL
# ============================
model_name = "IMSyPP/hate_speech_multilingual"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ============================
# 4⃣ TẠO PIPELINE (CPU)
# ============================
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

# ============================
# 5⃣ MAP NHÃN VỀ SAFE / VIOLATION
# ============================
label_map = {
    "LABEL_0": "SAFE",        # APPROPRIATE
    "LABEL_1": "VIOLATION",   # INAPPROPRIATE
    "LABEL_2": "VIOLATION",   # OFFENSIVE
    "LABEL_3": "VIOLATION"    # VIOLENT
}

# ============================
# 6⃣ TEST NHIỀU NGÔN NGỮ
# ============================
samples = [
    "Địt mẹ mày!",                  # Vietnamese offensive
    "Tôi ghét mấy người đó!",       # Vietnamese hate
    "Mấy thằng cali",                # Slang
    "Chế độ độc tài!",               # Political
    "Je t'adore mon ami."            # French positive
]

# ============================
# 7⃣ IN KẾT QUẢ
# ============================
results = clf(samples)

for text, res in zip(samples, results):
    label = label_map.get(res["label"], res["label"])
    print(f"{text}\n → {label} (score={res['score']:.4f})\n")
