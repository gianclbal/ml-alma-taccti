import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_for_theme(model_name_or_path):
    try:
        # Load the tokenizer and model for the given theme from Hugging Face Model Hub
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        model.eval()

    except Exception as e:
        print(f"[ERROR] Failed to load model from {model_name_or_path}: {e}")
        return None

    def predictor(sentences, batch_size=16):
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

            for sent, pred, prob in zip(batch, preds, probs):
                results.append({
                    "sentence": sent,
                    "prediction": int(pred.item()),
                    "probabilities": prob.tolist()
                })
        return results

    return predictor
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification


# def load_model_for_theme(model_path):
#     try:
#         # Load the tokenizer and model for the given theme
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         model.eval()

#         # Save the tokenizer to a local directory (optional)
#         tokenizer.save_pretrained(model_path)  # Save tokenizer in the same directory as model

#     except Exception as e:
#         print(f"[ERROR] Failed to load model from {model_path}: {e}")
#         return None

#     def predictor(sentences, batch_size=16):
#         results = []
#         for i in range(0, len(sentences), batch_size):
#             batch = sentences[i:i+batch_size]
#             inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 logits = outputs.logits
#                 probs = torch.softmax(logits, dim=1)
#                 preds = torch.argmax(probs, dim=1)

#             for sent, pred, prob in zip(batch, preds, probs):
#                 results.append({
#                     "sentence": sent,
#                     "prediction": int(pred.item()),
#                     "probabilities": prob.tolist()
#                 })
#         return results

#     return predictor

def predict_sentences(sentences, model_path):
    # Load the model dynamically using the theme's model path
    predictor = load_model_for_theme(model_path)
    if predictor is None:
        return []

    # Use the predictor for prediction
    return predictor(sentences)