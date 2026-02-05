import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "emotion_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

emotion_labels = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("Emotion Detection from Text")
st.write("Enter a sentence to detect the emotion")

text = st.text_area("Enter text here:")

if st.button("Analyze Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

        st.success(f"Emotion: **{emotion_labels[predicted_class.item()]}**")
        st.info(f"Confidence: **{confidence.item():.2f}**")
