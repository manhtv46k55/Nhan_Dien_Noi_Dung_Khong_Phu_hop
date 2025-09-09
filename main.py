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
    "LABEL_0": "Không vi phạm", # APPROPRIATE
    "LABEL_1": "Vi phạm",   # INAPPROPRIATE
    "LABEL_2": "Vi phạm",   # OFFENSIVE
    "LABEL_3": "Vi phạm"    # VIOLENT
}

# ============================
# 6⃣ TEST NHIỀU NGÔN NGỮ
# ============================
# Danh sách câu để test
samples = [
    # Vietnamese offensive / hate speech
    "Địt mẹ mày!",
    "Tôi ghét mấy người đó!",
    "Mày là đồ ngu!",
    "Bọn chó khốn nạn kia!",
    "Mấy thằng cali",
    "Đồ phản quốc, lũ súc vật!",

    # Vietnamese neutral / political / sensitive
    "Chế độ độc tài!",
    "Đảng này chỉ toàn tham nhũng.",
    "Tôi không quan tâm đến chính trị.",
    "Hôm nay trời rất đẹp.",

    # English hate/offensive
    "You are such an idiot!",
    "I hate people like you.",
    "Go back to your country!",
    "That group is full of trash.",

    # English neutral/positive
    "I love my family.",
    "What a wonderful day!",
    "He is working very hard to succeed.",
    "Let's go play football tomorrow.",

    # French (positive / neutral)
    "Je t'adore mon ami.",
    "Il fait très beau aujourd'hui.",
    "C'est une mauvaise idée.",

    # Mixed / tricky
    "Mày là đồ chó nhưng tao vẫn quý mày.",  # offensive + friendly
    "Thằng đó chơi ngu thật.",              # slang insult
    "I hate racism, but I love diversity.", # contains 'hate' but positive context
    "Stop spreading fake news!",            # political tone
]


# ============================
# 7⃣ IN KẾT QUẢ
# ============================
results = clf(samples)

for text, res in zip(samples, results):
    label = label_map.get(res["label"], res["label"])
    print(f"{text}\n → {label} (score={res['score']:.4f})\n")
