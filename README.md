# ProtClassifier — Protein Function Predictor

An app for predicting protein molecular functions using ProtBERT + MLP.

## Features
- Uses **ProtBERT** for embedding protein sequences
- Classifies into Binding / Enzyme / Transport / Membrane / Other
- GUI support for live prediction with Tkinter
- Accuracy and F1-score reported after training

## How to Run

The first run may take a long time, You can use the data in test_protein.txt to test.

```bash
python predict.py
```


## Directory Structure

├── data.tsv                   # Original raw sequence data
├── labeled_data.csv           # Labeled data with function categories
├── model_def.py               # MLP model definition
├── mlp_model.pth              # Trained model weights
├── predict.py                 # GUI application for protein function prediction
├── project.ipynb              # Main notebook (model training and evaluation)
├── high_comparison.ipynb      # Baseline: TF-IDF + Logistic Regression
├── low_comparation.ipynb      # Baseline: Amino Acid Count + Logistic Regression
├── protbert_embeddings.npy    # Cached BERT embeddings
├── test_protein.txt           # Sample input for testing GUI
├── README.md                  # Project introduction and usage guide


