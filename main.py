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
    # --- Offensive / Hate (VN) ---
    "Địt mẹ mày!",
    "Tôi ghét mấy người đó!",
    "Mày là đồ ngu!",
    "Bọn chó khốn nạn kia!",
    "Mấy thằng cali",
    "Đồ phản quốc, lũ súc vật!",
    "Bọn tây lông đấy chỉ biết ăn chơi.",
    "Lũ Ấn Độ bẩn thỉu.",
    "Người da đen toàn là tội phạm.",
    "Bọn đồng tính là bệnh hoạn.",
    "Mấy thằng lùn xấu xí.",
    "Đám người đó toàn rác rưởi.",
    "Tao thề mày ngu như bò.",
    "Cút về nước mày đi!",
    "Đám đó chỉ biết ăn bám xã hội.",
    "Mày đúng là thằng vô dụng.",
    "Tao không thèm nói chuyện với mày nữa.",
    "Bạn là người thông minh mà, bạn hiểu tôi đang nói gì đúng không? suy cho cùng thì chế độ này vẫn đầy rẫy cạm bẫy.",
    "Toàn dân hãy đứng lên chống lại bọn cướp nước!",
    "Bọn tham nhũng chỉ biết vơ vét của dân.",
    "Mày đúng là thằng khốn nạn.",
    "Chính quyền bù nhìn",
    "Bọn chúng mày chỉ biết ăn chơi trác táng.",
    "Mày đúng là đồ chó đẻ.",
    "Tao sẽ cho mày biết thế nào là đau đớn.",
    "Mày đúng là thằng mất dạy.",
    "Chính quyền Việt Nam là bọn tay sai cho Trung Quốc.",
    "Bọn chúng mày đúng là lũ súc vật.",
    "Mày đúng là thằng ăn bám xã hội.",

    # --- Neutral / Political (VN) ---
    "Chế độ độc tài!",
    "Đảng này chỉ toàn tham nhũng.",
    "Tôi không quan tâm đến chính trị.",
    "Hôm nay trời rất đẹp.",

    # --- English Offensive ---
    "You are such an idiot!",
    "I hate people like you.",
    "Go back to your country!",
    "That group is full of trash.",

    # --- English Neutral / Positive ---
    "I love my family.",
    "What a wonderful day!",
    "He is working very hard to succeed.",
    "Let's go play football tomorrow.",

    # --- French ---
    "Je t'adore mon ami.",
    "Il fait très beau aujourd'hui.",
    "C'est une mauvaise idée.",

    # --- Mixed / Slang ---
    "Mày là đồ chó nhưng tao vẫn quý mày.",
    "Thằng đó chơi ngu thật.",
    "I hate racism, but I love diversity.",
    "Stop spreading fake news!",

    # --- 18+ / NSFW ---
    "Tao muốn làm tình với mày.",
    "Địt con mẹ nó phê vãi.",
    "Mày chỉ giỏi khoe hàng thôi.",
    "Thằng kia toàn xem phim sex.",
    "This porn video is disgusting.",
    "She has a sexy body, I want her now.",
    "Anh muốn ngủ với em tối nay.",
    "Cái váy này nhìn gợi cảm quá.",
]


# ============================
# 7⃣ IN KẾT QUẢ
# ============================
results = clf(samples)

for text, res in zip(samples, results):
    label = label_map.get(res["label"], res["label"])
    print(f"{text}\n → {label} (score={res['score']:.4f})\n")
