import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.preprocessing import normalize
from model_def import MLP

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model structure and parameters
model = MLP().to(device)
model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
model.eval()

# Load the tokenizer and bert model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
bert_model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
bert_model.eval()

# Label Mapping Table）
id2label = {
    0: "Binding",
    1: "Enzyme",
    2: "Membrane",
    3: "Other",
    4: "Transport"
}

# preprocess function
def preprocess_sequence(seq):
    seq = seq.replace(" ", "")
    seq = ' '.join(list(seq))
    return f"[CLS] {seq} [SEP]"

# get embedding（mean pooling）
@torch.no_grad()
def get_embedding(seq):
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = bert_model(**inputs)
    last_hidden = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / summed_mask
    return mean_pooled.cpu().numpy()

def generate_description(label):
    descriptions = {
        "Binding": "This protein may be involved in molecular binding.",
        "Enzyme": "This protein may have catalytic functions, participating in biochemical reactions.",
        "Membrane": "This protein may be located on or within membranes, contributing to structure or signaling.",
        "Transport": "This protein may be responsible for transporting ions, molecules, or energy across membranes.",
        "Other": "The function of this protein is unclear or does not fall into the predefined categories."
    }
    return descriptions.get(label, "Unknown function.")


# Prediction function
def predict():
    seq_input = input_text.get("1.0", tk.END).strip()
    if not seq_input:
        messagebox.showwarning("Warning", "Please enter a protein sequence.")
        return
    try:
        pre_seq = preprocess_sequence(seq_input)
        embedding = normalize(get_embedding(pre_seq), axis=1)

        # Avoid BatchNorm issues by padding to batch size 2 if needed
        if embedding.shape[0] == 1:
            dummy = np.zeros_like(embedding)
            embedding = np.vstack([embedding, dummy])
            input_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
            output = model(input_tensor)[0:1]
        else:
            input_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
            output = model(input_tensor)

        pred_id = torch.argmax(output, dim=1).item()
        pred_label = id2label[pred_id]
        description = generate_description(pred_label)

        result_var.set(f" Prediction: {pred_label}\n Description: {description}")
    except Exception as e:
        messagebox.showerror("error", str(e))


# UI
window = tk.Tk()
window.title("ProtClassifier: Protein Function Predictor")
window.geometry("600x400")

tk.Label(window, text="Enter a protein sequence:", font=("Arial", 12)).pack(pady=10)
input_text = tk.Text(window, height=5, width=70)
input_text.pack()

tk.Button(window, text="Predict", command=predict, font=("Arial", 12)).pack(pady=10)
result_var = tk.StringVar()
result_label = tk.Label(window, textvariable=result_var, font=("Arial", 12), justify="left", wraplength=550)
result_label.pack(pady=10)

window.mainloop()