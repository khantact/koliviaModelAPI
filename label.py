from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, evaluate, accelerate
from transformers import TrainingArguments, Trainer
import glob
import numpy as np
from datasets import Dataset, load_dataset
import optuna
import random

app = Flask(__name__)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


@app.route("/predict", methods=["GET"])
def predict():
    return getScore("make an appointment for 2pm on friday")


@torch.no_grad()
def getScore(review: str):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-classifier", num_labels=3
        ).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-classifier", use_fast=True
        )

        model.eval()
        model = model.to("cpu")
        input_ids = tokenizer(
            review, return_tensors="pt", padding=True, truncation=True
        )
        output = model(**input_ids).logits
        pred = np.argmax(output, axis=-1).tolist()[0]

        categories = ["Appointments", "Questions", "Weather"]
        return categories[pred]
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=105)
