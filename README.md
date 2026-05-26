# ProtClassifier — Protein Function Predictor

An academic machine learning project for predicting broad protein molecular functions from amino acid sequences using ProtBERT embeddings and a PyTorch MLP classifier.

## Features
- Uses ProtBERT to generate protein sequence embeddings
- Classifies protein sequences into broad functional categories: Binding, Enzyme, Transport, Membrane, and Other
- Includes notebooks for model development and comparison experiments
- Provides a sample protein sequence input for testing

## Tech Stack
Python, PyTorch, HuggingFace Transformers, ProtBERT, scikit-learn, pandas, NumPy, Tkinter

## Dataset
The dataset was constructed from UniProt protein records and function descriptions. A labeled dataset of 17,421 protein samples was created across five functional categories.

Raw dataset files and trained model checkpoints are not included due to file size and data source considerations.

## Repository Contents
- `project.ipynb`: main notebook for data processing, embedding extraction, training, and evaluation
- `high_comparison.ipynb`: comparison experiment notebook
- `low_comparison.ipynb`: baseline comparison notebook
- `model_def.py`: PyTorch MLP model definition
- `predict.py`: prediction script for protein function classification
- `test_protein.txt`: sample protein sequence input
- `README.md`: project overview and usage guide

## How to Run
```bash
python predict.py
