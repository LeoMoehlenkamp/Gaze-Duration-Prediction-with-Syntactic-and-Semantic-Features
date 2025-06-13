#  Gaze Duration Prediction – Syntax vs. Semantics

This project explores how well gaze duration during reading can be predicted using different types of linguistic features. Using data from the **GECO corpus**, the goal is to compare the predictive power of syntactic versus semantic information.

##  Project Goal

- Predict the gaze duration of words during first-pass reading  
- Compare models based on syntactic vs. semantic input features  
- Evaluate whether combining both improves prediction performance

##  Models

### Model A – Linear Regression (Syntactic Features)
A simple, transparent baseline model using:
- Word length  
- Position in sentence  
- Word frequency  

### Model B – Neural Network with GPT Embeddings
A more complex model using:
- GPT-based word embeddings via OpenAI API  
- Captures deep semantic meaning and context

### (Optional) Model C – Hybrid Model
A combined model integrating both syntactic and semantic features into one neural network.

##  Hypotheses

- **H1:** Semantic embeddings improve prediction of gaze duration  
- **H2:** Syntactic features already explain a substantial part of the variance

---

## ⚠️ Note on Large Files

The file `MonolingualReadingData.csv` (≈144 MB) is **not included** in this repository due to GitHub's file size limitations.

To run the code, please manually place the file in the following path:

Data/MonolingualReadingData.csv

Download MonolingualReadingData.xlsx here: https://expsy.ugent.be/downloads/geco/
