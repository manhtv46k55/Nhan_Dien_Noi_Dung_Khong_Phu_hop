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
    "LABEL_0": "Không vi phạm",  # APPROPRIATE
    "LABEL_1": "Vi phạm",  # INAPPROPRIATE
    "LABEL_2": "Vi phạm",  # OFFENSIVE
    "LABEL_3": "Vi phạm",  # VIOLENT
}


# ============================
# 6⃣ TẠO LỚP FILTER CHO OPEN-WEBUI
# ============================
class MessageFilter:
    def __init__(self):
        self.clf = clf
        self.label_map = label_map

    def check_message(self, message: str) -> dict:
        """
        Kiểm tra tin nhắn:
        - Nếu vi phạm: trả ngay kết quả {"violation": True, "label": "...", "score": ...}
        - Nếu không vi phạm: trả {"violation": False}
        """
        result = self.clf(message)[0]
        label = self.label_map.get(result["label"], result["label"])
        score = result["score"]

        if label == "Vi phạm":
            return {"violation": True, "label": label, "score": score}
        else:
            return {"violation": False}


# ============================
# 7⃣ TÍCH HỢP VÀO INLET CỦA OPEN-WEBUI
# ============================
class Filter:
    def __init__(self):
        self.msg_filter = MessageFilter()

    def inlet(self, body: dict, __user__: dict = None) -> dict:
        messages = body.get("messages", [])
        for msg in messages:
            text = msg.get("content", "")
            check = self.msg_filter.check_message(text)
            if check["violation"]:
                # Thay vì raise Exception, replace nội dung
                msg["content"] = (
                    "Nội dung này đã vi phạm quy định chung, vui lòng không trả lời."
                )
        return body

    def outlet(self, body: dict, __user__: dict = None) -> dict:
        return body
