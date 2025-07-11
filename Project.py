#!/usr/bin/env python
# coding: utf-8

# ___
# ### Import data, all needed packages and display data

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

nltk.download("brown")
from nltk.corpus import brown
from collections import Counter
pd.set_option("display.max_rows", 100)
pd.set_option("display.expand_frame_repr", False)


# In[2]:


import torch
print(torch.cuda.is_available())           # → True
print(torch.cuda.get_device_name(0))       # → NVIDIA GeForce RTX 4060 Ti


# In[3]:


use_cols = ["WORD", "WORD_GAZE_DURATION"]
df = pd.read_csv("Data\MonolingualReadingData.csv", usecols=use_cols)
print(df.head())


# ___
# ### Create new columns that are needed for later modeling

# In[4]:


# find end of sentence
df["is_sentence_end"] =  df["WORD"].str.contains(r"[.!?]$", regex=True)
df["is_sentence_end"] = df["WORD"].str.contains(r"[.!?][\"')]*$", regex=True)


# find start of sentence
df["is_sentence_start"] = df["is_sentence_end"].shift(1, fill_value=False)

# make first line the start of a sentence
df.loc[0, "is_sentence_start"] = True

# sentence id is incremented by sentence start
df["sentence_id"] = df["is_sentence_start"].cumsum()
df.loc[0, "sentence_id"] = 1

# calculate position in sentence
df["word_pos_in_sentence"] = df.groupby("sentence_id").cumcount() + 1


# In[5]:


print(df[["is_sentence_start", "is_sentence_end"]].head(20))


# In[6]:


print(df[["WORD", "is_sentence_end", "sentence_id", "word_pos_in_sentence"]].tail(20))


# ___
# ### Import text from experiment and nltk corpus for global and local frequency calculation

# In[7]:


with open("Data\Corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:500])


# In[8]:


brown_tokens = [w.lower() for w in brown.words() if w.isalpha()]
text_tokens = re.findall(r"\b\w+\b", text.lower())


# In[9]:


brown_counter = Counter(brown_tokens)
text_counter = Counter(text_tokens)
total_brown = sum(brown_counter.values())
total_text = sum(text_counter.values())

print(brown_counter)
print(text_counter)


# ___
# ### Calculation of global and local frequencies

# In[10]:


df["word_lower"] = df["WORD"].str.replace(r"[^a-zA-Z]", "", regex=True).str.lower()
df["global_freq_abs"] = df["word_lower"].map(brown_counter).fillna(0)
df["local_freq_abs"] = df["word_lower"].map(text_counter).fillna(0)
df["global_freq_rel"] = df["global_freq_abs"]/total_brown
df["local_freq_rel"] = df["local_freq_abs"]/total_text
df["log_global_rel"] = np.log1p(df["global_freq_rel"])
df["log_local_rel"] = np.log1p(df["local_freq_rel"])
print(df[["WORD", "word_lower", "global_freq_abs", "local_freq_abs", "global_freq_rel", "local_freq_rel", "log_global_rel", "log_local_rel"]].tail(20))


# In[11]:


df[["log_global_rel", "log_local_rel"]].corr()


# ___
# ### Adding word length column

# In[12]:


df["word_length"] = df["word_lower"].str.len()
print(df[["word_lower", "word_length", "word_pos_in_sentence", "log_global_rel", "log_local_rel", "WORD_GAZE_DURATION"]].tail(20))


# ___
# ### Linear model using grid search (unvectorized and then vectorized for better performance)

# In[13]:


df_clean = df[df["WORD_GAZE_DURATION"].apply(lambda x: str(x).isdigit())].copy()
df_clean["WORD_GAZE_DURATION"] = df_clean["WORD_GAZE_DURATION"].astype(float)
df_clean = df_clean.dropna(subset=["word_lower"]).reset_index(drop=True)


# In[14]:


def predict_gaze_duration(row, weights, bias):
    return (
        weights[0] * row["word_length"]
        + weights[1] * row["word_pos_in_sentence"]
        + weights[2] * row["log_global_rel"]
        + bias
    )

def grid_search(param_range, stepsize, init_weights, bias):
    best_mse = float("inf")
    best_weights = init_weights
    mse_list = []

    steps = int(param_range / stepsize)
    offset = param_range / 2

    for i in range(steps):
        alpha = init_weights[0] - offset + i * stepsize
        for j in range(steps):
            beta = init_weights[1] - offset + j * stepsize
            for k in range(steps):
                gamma = init_weights[2] - offset + k * stepsize
                weights = [alpha, beta, gamma]
                
                df_clean["prediction"] = df_clean.apply(
                    lambda row: predict_gaze_duration(row, weights, bias), axis=1
                )
                mse = np.mean((df_clean["prediction"] - df_clean["WORD_GAZE_DURATION"])**2)
                mse_list.append((mse, weights))

                if mse < best_mse:
                    best_mse = mse
                    best_weights = weights

    return best_weights, best_mse, mse_list


# In[15]:


init_weights = np.random.uniform(-10, 10, size=3).tolist()
init_bias = np.random.uniform(-10, 10)

best_weights, best_mse, _ = grid_search(2, 0.5, init_weights, init_bias)

print("Best Weights:", best_weights)
print("Best MSE:", best_mse)


# > **Performance Warning**  
# This pure-Python grid search loops over every row for each weight combo—extremely slow. We’ll switch to a fully vectorized approach instead.
# ___

# In[16]:


df_clean = df_clean.dropna(subset=["word_length", "word_pos_in_sentence", "log_global_rel", "WORD_GAZE_DURATION"]).copy()

X = df_clean[["word_length", "word_pos_in_sentence", "log_global_rel"]].values
y = df_clean["WORD_GAZE_DURATION"].values


def predict_gaze_duration(X, weights, bias):
    return X @ weights + bias

def grid_search(X, y, param_range, stepsize, init_weights, init_bias):
    best_mse = float("inf")
    best_weights = init_weights
    best_bias = init_bias
    mse_list = []

    steps = int(param_range / stepsize)
    offset = param_range / 2

    for i in range(steps):
        alpha = init_weights[0] - offset + i * stepsize
        for j in range(steps):
            beta = init_weights[1] - offset + j * stepsize
            for k in range(steps):
                gamma = init_weights[2] - offset + k * stepsize
                for b in range(steps):
                    bias = init_bias - offset + b * stepsize
                    weights = np.array([alpha, beta, gamma])

                    preds = X @ weights + bias
                    mse = np.mean((preds - y) ** 2)
                    mse_list.append((mse, weights, bias))

                    if mse < best_mse:
                        best_mse = mse
                        best_weights = weights
                        best_bias = bias

    return best_weights, best_bias, best_mse, mse_list


# In[23]:


# Settings: (param_range, stepsize)
refinements = [(20, 2), (5, 0.5), (2, 0.2)]

# Initialisation
init_weights = np.random.uniform(-10, 10, size=3).tolist()
init_bias    = np.random.uniform(-10, 10)

# Liste zum Sammeln der Results
results = []  # wird (param_range, best_mse)

for pr, ss in refinements:
    best_weights, best_bias, best_mse, _ = grid_search(
        X, y,
        param_range=pr,
        stepsize=ss,
        init_weights=init_weights,
        init_bias=init_bias
    )
    print(f"Range={pr}, Step={ss} → Best MSE: {best_mse:.2f}")
    results.append((pr, best_mse))

    # für die nächste Verfeinerung
    init_weights = best_weights
    init_bias    = best_bias

# --- Plot nach der Schleife ---
param_ranges = [r for r, _ in results]
best_mses    = [m for _, m in results]

runs = list(range(1, len(best_mses) + 1))

plt.figure(figsize=(6, 4))
plt.plot(runs, best_mses, marker='o', linewidth=2)
plt.xticks(runs, [f"Run {i}" for i in runs])
plt.xlabel("Grid Search Run")
plt.ylabel("Best MSE")
plt.title("Best MSE per Run")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# > **Performance Warning**  
# This is still extremely slow. We’ll switch to a fully gradient descent-based approach instead.

# ___
# ### Comparison with Gradient Descent-Based Regression Model

# In[24]:


def predict_gaze_duration(X, weights, bias):
    return X @ weights + bias

def gradient_descent(X, y, init_weights, init_bias, learning_rate=1e-6, epochs=1000):
    weights = np.array(init_weights, dtype=np.float64)
    bias = init_bias
    n = len(y)
    mse_list = []

    for epoch in range(epochs):
        preds = predict_gaze_duration(X, weights, bias)
        errors = preds - y

        # Gradienten berechnen
        gradient_w = (2/n) * (X.T @ errors)
        gradient_b = (2/n) * np.sum(errors)

        # Parameter-Update
        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b

        # MSE berechnen
        mse = np.mean(errors ** 2)
        mse_list.append(mse)

        # Optional: abbrechen, wenn Änderung sehr klein
        if epoch > 10 and abs(mse_list[-2] - mse_list[-1]) < 1e-5:
            break

    return weights, bias, mse_list[-1], mse_list, preds


X = df_clean[["word_length", "word_pos_in_sentence", "log_global_rel"]].values
y = df_clean["WORD_GAZE_DURATION"].values

init_weights = np.random.uniform(-10, 10, size=3).tolist()
init_bias = np.random.uniform(-10, 10)
weights, bias, best_mse, mse_list, preds = gradient_descent(X, y, init_weights, init_bias, learning_rate=0.005, epochs=3000)

print("Best Weights:", best_weights)
print("Best bias:", best_bias)
print("Best MSE:", best_mse)
print("Best RMSE (ms):", np.sqrt(best_mse), "ms")


# In[25]:


plt.figure(figsize=(10, 5))
plt.plot(mse_list, label="MSE per Epoch", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("MSE over Epochs during Gradient Descent")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


comparison_df = pd.DataFrame({
    "word": df_clean["word_lower"].values,
    "gaze_duration_actual": y,
    "gaze_duration_predicted": preds
})


# ___
# ## GPT embeddings and ANN
# 
# #### The following cell fetches GPT embeddings for all unique words and saves them to a local cache file (`embeddings_dict.pkl`). On later runs, there is <span style="color:red">no need to run the following cell</span>.  
# 

# In[56]:


from openai import OpenAI
import time
import openai
import pickle
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#Add api key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# In[ ]:


from openai import OpenAI
import time
import openai
import pickle
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#Add api key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#function to get embeddings
def get_embedding(word, retries=5, wait=1):
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=word
            )
            return response.data[0].embedding
        except openai.RateLimitError:
            print(f"Rate Limit erreicht bei '{word}' – Versuch {attempt + 1}, warte {wait}s...")
            time.sleep(wait)
            wait *= 2
        except Exception as e:
            print(f"Fehler bei Wort '{word}': {e}")
            return None
    print(f"Fehlgeschlagen nach {retries} Versuchen: '{word}'")
    return None


#Getting all the unique words to spare api requests
unique_words = df_clean["word_lower"].dropna().unique()

#get embeddings for every word
embedding_dict = {}

for i, word in enumerate(unique_words):
    embedding = get_embedding(word)
    if embedding:
        embedding_dict[word] = embedding
    time.sleep(0.1)


#save the dictionary as pkl file
with open("embeddings_dict.pkl", "wb") as f:
    pickle.dump(embedding_dict, f)


# ### Load the pkl file that contains the embeddings

# In[26]:


import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import pickle
import torch.nn as nn

unique_words = df_clean["word_lower"].dropna().unique()

#Load and check data
with open("embeddings_dict.pkl", "rb") as f:
    embedding_dict = pickle.load(f)


# ### Prepare Embeddings, Transform Target & Create DataLoaders

# In[28]:


df_clean["embedding"] = df_clean["word_lower"].map(embedding_dict)

X_embed = np.vstack(df_clean["embedding"].values)
y       = df_clean["WORD_GAZE_DURATION"].values  

df_clean["log_gaze"] = np.log1p(df_clean["WORD_GAZE_DURATION"])

y = torch.from_numpy(df_clean["log_gaze"].values).float().unsqueeze(1)
X = torch.from_numpy(X_embed).float()
# Dataset
ds = TensorDataset(X, y)

# Größen berechnen
n = len(ds)
n_train = int(0.8 * n)
n_temp  = n - n_train
n_val   = n_temp // 2
n_test  = n_temp - n_val

# Zufällige Aufteilung mit festem Seed
train_ds, val_ds, test_ds = random_split(
    ds,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

# DataLoader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")


# ### Define & Train Neural Network Model

# In[29]:


# --- Model Definition ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class GazeNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

model_semantic = GazeNet(input_dim=X.shape[1]).to(device)

# --- Loss & Optimizer ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_semantic.parameters(), lr=1e-3)

# --- Training Loop with Validation ---
n_epochs = 50
train_mses = []
val_mses   = []

for epoch in range(1, n_epochs+1):
    # Training
    model_semantic.train()
    total_train = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model_semantic(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_train += loss.item() * Xb.size(0)
    train_mse = total_train / len(train_loader.dataset)
    train_mses.append(train_mse)

    # Validation
    model_semantic.eval()
    total_val = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model_semantic(Xb)
            total_val += criterion(preds, yb).item() * Xb.size(0)
    val_mse = total_val / len(val_loader.dataset)
    val_mses.append(val_mse)

    print(f"Epoch {epoch:02d}/{n_epochs} — Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")

# --- Plotting Train vs. Validation MSE ---
epochs = list(range(1, n_epochs+1))
plt.figure(figsize=(8,5))
plt.plot(epochs, train_mses, label="Train MSE", marker='o')
plt.plot(epochs, val_mses,   label="Val   MSE", marker='s')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training vs. Validation MSE per Epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[30]:


torch.save(model_semantic.state_dict(), "model_semantic_weights.pth")
print("model_semantic-weights saved")


# ### Evaluate Model & Compute Final MSE/RMSE

# In[31]:


# Modell in Eval-Mode
model_semantic.eval()
preds_log, trues_log = [], []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out = model_semantic(Xb)
        preds_log.append(out.cpu().numpy())
        trues_log.append(yb.cpu().numpy())

# Arrays zusammenführen
preds_log = np.vstack(preds_log).ravel()
trues_log = np.vstack(trues_log).ravel()

# Log-MSE
mse_log = np.mean((preds_log - trues_log)**2)

# Rücktransform und Original-MSE/RMSE
preds_ms = np.expm1(preds_log)
trues_ms = np.expm1(trues_log)
mse_ms  = np.mean((preds_ms - trues_ms)**2)
rmse_ms = np.sqrt(mse_ms)

print(f"Final Test MSE (log scale): {mse_log:.4f}")
print(f"Final Test MSE (ms):        {mse_ms:.4f}")
print(f"Final Test RMSE (ms):       {rmse_ms:.1f} ms")


# ___
# ### Combine Semantic & Syntactic Features

# In[32]:


df_clean["embedding"] = df_clean["word_lower"].map(embedding_dict)

X_embed = np.vstack(df_clean["embedding"].values)
X_syntax = df_clean[["word_length","word_pos_in_sentence","log_global_rel"]].values
X_comb = np.hstack([X_embed, X_syntax])

df_clean["log_gaze"] = np.log1p(df_clean["WORD_GAZE_DURATION"])

y = torch.from_numpy(df_clean["log_gaze"].values).float().unsqueeze(1)
X = torch.from_numpy(X_comb).float()
ds = TensorDataset(X, y)

n     = len(ds)
n_train = int(0.8 * n)
n_temp  = n - n_train
n_val   = n_temp // 2
n_test  = n_temp - n_val 


train_ds, val_ds, test_ds = random_split(
    ds,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")


# ### Define & Train Neural Network Model

# In[33]:


# --- Model & Device wie gehabt ---
model_combined = GazeNet(input_dim=X.shape[1]).to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_combined.parameters(), lr=1e-3)

# --- Training Loop mit Validation ---
n_epochs = 50
train_mses = []
val_mses   = []

for epoch in range(1, n_epochs+1):
    # --- Training ---
    model_combined.train()
    total_train = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model_combined(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_train += loss.item() * Xb.size(0)
    train_mse = total_train / len(train_loader.dataset)
    train_mses.append(train_mse)

    # --- Validation auf val_loader ---
    model_combined.eval()
    total_val = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model_combined(Xb)
            total_val += criterion(preds, yb).item() * Xb.size(0)
    val_mse = total_val / len(val_loader.dataset)
    val_mses.append(val_mse)

    print(f"Epoch {epoch:02d}/{n_epochs} — Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")

# --- Plot Training vs. Validation MSE ---
import matplotlib.pyplot as plt

epochs = list(range(1, n_epochs+1))
plt.figure(figsize=(8,5))
plt.plot(epochs, train_mses, label="Train MSE", marker='o')
plt.plot(epochs, val_mses,   label="Val   MSE", marker='s')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Combined Model — Train vs. Validation MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[34]:


torch.save(model_combined.state_dict(), "model_combined_weights.pth")
print("model_combined-weights saved")


# ### Evaluate Model & Compute Final MSE/RMSE

# In[35]:


# Modell in Eval-Mode
model_combined.eval()
preds_log, trues_log = [], []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out = model_combined(Xb)
        preds_log.append(out.cpu().numpy())
        trues_log.append(yb.cpu().numpy())

# Arrays zusammenführen
preds_log = np.vstack(preds_log).ravel()
trues_log = np.vstack(trues_log).ravel()

# Log-MSE
mse_log = np.mean((preds_log - trues_log)**2)

# Rücktransform und Original-MSE/RMSE
preds_ms = np.expm1(preds_log)
trues_ms = np.expm1(trues_log)
mse_ms  = np.mean((preds_ms - trues_ms)**2)
rmse_ms = np.sqrt(mse_ms)

print(f"Final Test MSE (log scale): {mse_log:.4f}")
print(f"Final Test MSE (ms):        {mse_ms:.4f}")
print(f"Final Test RMSE (ms):       {rmse_ms:.1f} ms")


# ___

# In[ ]:


EMBED_DIM = len(next(iter(embedding_dict.values())))
def load_model(path, input_dim):
    m = GazeNet(input_dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

MODEL_SEMANTIC = load_model("model_semantic_weights.pth", EMBED_DIM)
MODEL_COMBINED = load_model("model_combined_weights.pth", EMBED_DIM + 3)

# Preprocessing Functions
def preprocess_semantic(sentence):
    X = []
    for w in sentence:
        wl = w.lower()
        X.append(embedding_dict.get(wl, np.zeros(EMBED_DIM)))
    return torch.from_numpy(np.vstack(X)).float().to(device)

def preprocess_combined(sentence):
    X_sem, X_syn = [], []
    for idx, w in enumerate(sentence):
        wl = w.lower()
        emb = embedding_dict.get(wl, np.zeros(EMBED_DIM))
        length   = len(wl)
        position = idx + 1
        log_glob = np.log1p(brown_counter[wl] / total_brown)
        X_sem.append(emb)
        X_syn.append([length, position, log_glob])
    X_sem = np.vstack(X_sem)
    X_syn = np.vstack(X_syn)
    return torch.from_numpy(np.hstack([X_sem, X_syn])).float().to(device)

# Prediction Wrappers
def predict_semantic(sentence):
    X = preprocess_semantic(sentence)
    with torch.no_grad():
        preds_log = MODEL_SEMANTIC(X).cpu().numpy().ravel()
    return np.expm1(preds_log)

def predict_combined(sentence):
    X = preprocess_combined(sentence)
    with torch.no_grad():
        preds_log = MODEL_COMBINED(X).cpu().numpy().ravel()
    return np.expm1(preds_log)

# Example Usage
sentence = ["This", "is", "an", "example", "Sentence", "."]
print("Semantic-only preds:     ", predict_semantic(sentence))
print("Semantic+Syntax preds:   ", predict_combined(sentence))


# In[ ]:


# Extract first sentence from your cleaned DataFrame
first_sent = df_clean[df_clean["sentence_id"] == 1].reset_index(drop=True)
words       = first_sent["WORD"].tolist()
actual_ms   = first_sent["WORD_GAZE_DURATION"].tolist()

# Get predictions (semantic+syntax)
predicted_ms_semantic = predict_semantic(words)
predicted_ms_combined = predict_combined(words)

df_compare = pd.DataFrame({
    "word":      words,
    "actual_ms": actual_ms,
    "pred_ms_semantic":   np.round(predicted_ms_semantic, 1),
    "predicted_ms_combined":   np.round(predicted_ms_combined, 1)
})

print(df_compare)


# In[38]:


from sklearn.decomposition import PCA
import numpy as np
df_unique = df_clean.drop_duplicates(subset="word_lower").copy()
X_embed = np.vstack(df_unique["embedding"].values)
# 1) PCA ohne Dim-Limit fitten (oder n_components=X_embed.shape[1])
pca_full = PCA().fit(X_embed)

# 2) Kumulierte Varianz berechnen
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# 3) Index finden, ab dem ≥80 %
n_80 = np.searchsorted(cum_var, 0.8) + 1
print(f"Anzahl PCs für ≥80 % Varianz: {n_80}")
print(f"Kumulierte Varianz bei PC{n_80}: {cum_var[n_80-1]:.3f}")


# ___
# ### From here on all code can be skipped, since the data generated here is saved as df_clusters.pkl. Run the code where it says "RUN FROM HERE"

# In[63]:


df_unique = df_clean.drop_duplicates(subset="word_lower").copy()
X_embed = np.vstack(df_unique["embedding"].values)

# 2) PCA auf 142 Komponenten (80 % Varianz)
pca = PCA(n_components=236)
X_pca = pca.fit_transform(X_embed)

# 3) KMeans-Clustering im 142-dimensionalen Raum
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_pca)

# 4) 3D-Scatterplot der ersten drei PCs
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
    c=labels, s=10, alpha=0.6
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA (236D) – 3D Plot of PCs 1–3 with 5 clusters")
plt.tight_layout()
plt.show()

# 5) Explained Variance Ratio für die ersten 10 PCs
for i, ratio in enumerate(pca.explained_variance_ratio_[:10], start=1):
    print(f"PC{i}: {ratio:.4f}")
print(f"... (insgesamt {len(pca.explained_variance_ratio_)} PCs)")


# In[47]:


# Cluster direkt zu df_unique hinzufügen
df_unique["cluster"] = labels

# Für jeden Cluster die 10 häufigsten Wörter (hier in df_unique natürlich alle einmal)
for i in range(5):
    words_i = df_unique.loc[df_unique["cluster"] == i, "word_lower"]
    print(f"Cluster {i}: ", words_i.value_counts().head(10).index.tolist())


# In[48]:


pca = PCA(n_components=236)
X_pca = pca.fit_transform(X_embed)

# 3) KMeans-Clustering im 142-dimensionalen Raum
kmeans = KMeans(n_clusters=100, random_state=42)
labels = kmeans.fit_predict(X_pca)

df_unique["cluster"] = labels

# Für jeden Cluster die 10 häufigsten Wörter (hier in df_unique natürlich alle einmal)
for i in range(100):
    words_i = df_unique.loc[df_unique["cluster"] == i, "word_lower"]
    print(f"Cluster {i}: ", words_i.value_counts().head(10).index.tolist())


# In[51]:


word2cluster = dict(zip(df_unique["word_lower"], labels))

df_clean["cluster"] = df_clean["word_lower"].map(word2cluster)


# In[57]:


# 1) Deine Cluster-Wörter vorbereiten, z.B. dict: {cluster_id: [w1, w2, ...], ...}
df_clean_clusters = df_clean.copy()
clusters = {}
for cl in sorted(df_clean_clusters["cluster"].unique()):
    words = df_clean_clusters.loc[df_clean_clusters.cluster==cl, "word_lower"].value_counts().head(10).index.tolist()
    clusters[cl] = words

# 2) Label-Generierung
labels = {}
for cl, words in clusters.items():
    prompt = (
        "Bitte gib mir **ein einziges englisches Wort**, das am besten diese Liste von Wörtern "
        f"beschreibt: {', '.join(words)}."
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher Assistent, der Cluster-Kategorien prägnant in nur einem Wort auf englisch zusammenfasst."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=6,
        n=1
    )
    label = resp.choices[0].message.content.strip().strip('"')
    labels[cl] = label

# 3) Labels ins DataFrame einfügen
df_clean_clusters["cluster_label"] = df_clean_clusters["cluster"].map(labels)

# 4) Kontrolle
for cl, lab in labels.items():
    top10 = clusters[cl]  # deine vorher ermittelten Top‐10-Wörter
    print(f"Cluster {cl} ({lab}): {top10}")


# In[ ]:


df_cluster = df_clean_clusters.copy()
n
df_cluster.to_pickle("df_cluster.pkl")


# ___
# ### RUN FROM HERE

# In[59]:


df_cluster = pd.read_pickle("df_cluster.pkl")


# In[75]:


X_all       = np.vstack(df_cluster["embedding"].values)  # (472133, original_dim)
X_pca_all   = pca.transform(X_all)                       # (472133, 236)

df_cluster["embedding"] = list(X_pca_all)


# In[84]:


# 1) PCA-Embeddings direkt aus df_clean holen
X_embed  = np.vstack(df_cluster["embedding"].values)      # (N, 236)

# 2) Syntax-Features
X_syntax = df_cluster[["word_length","word_pos_in_sentence","log_global_rel"]].values  # (N, 3)

# 3) Kombinierte Eingabe
X_comb   = np.hstack([X_embed, X_syntax])               # (N, 239)

# 4) Zielwerte
df_cluster["log_gaze"] = np.log1p(df_cluster["WORD_GAZE_DURATION"])
y        = torch.from_numpy(df_cluster["log_gaze"].values).float().unsqueeze(1)


X_tensor = torch.from_numpy(X_comb).float()
ds        = TensorDataset(X_tensor, y)
n         = len(ds)
n_train   = int(0.8 * n)
n_temp    = n - n_train
n_val     = n_temp // 2
n_test    = n_temp - n_val

train_ds, val_ds, test_ds = random_split(
    ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# 6) Modell initieren mit input_dim=239
model_PCA = GazeNet(input_dim=X_comb.shape[1]).to(device)


# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_PCA.parameters(), lr=1e-3)

# --- Training Loop mit Validation ---
n_epochs = 50
train_mses = []
val_mses   = []

for epoch in range(1, n_epochs+1):
    # --- Training ---
    model_PCA.train()
    total_train = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model_PCA(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_train += loss.item() * Xb.size(0)
    train_mse = total_train / len(train_loader.dataset)
    train_mses.append(train_mse)

    # --- Validation auf val_loader ---
    model_PCA.eval()
    total_val = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model_PCA(Xb)
            total_val += criterion(preds, yb).item() * Xb.size(0)
    val_mse = total_val / len(val_loader.dataset)
    val_mses.append(val_mse)

    print(f"Epoch {epoch:02d}/{n_epochs} — Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")

# --- Plot Training vs. Validation MSE ---
import matplotlib.pyplot as plt

epochs = list(range(1, n_epochs+1))
plt.figure(figsize=(8,5))
plt.plot(epochs, train_mses, label="Train MSE", marker='o')
plt.plot(epochs, val_mses,   label="Val   MSE", marker='s')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Combined Model on PCA-Reduced Data — Train vs. Val. MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[85]:


# Modell in Eval-Mode
model_PCA.eval()
preds_log, trues_log = [], []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out = model_PCA(Xb)
        preds_log.append(out.cpu().numpy())
        trues_log.append(yb.cpu().numpy())

# Arrays zusammenführen
preds_log = np.vstack(preds_log).ravel()
trues_log = np.vstack(trues_log).ravel()

# Log-MSE
mse_log = np.mean((preds_log - trues_log)**2)

# Rücktransform und Original-MSE/RMSE
preds_ms = np.expm1(preds_log)
trues_ms = np.expm1(trues_log)
mse_ms  = np.mean((preds_ms - trues_ms)**2)
rmse_ms = np.sqrt(mse_ms)

print(f"Final Test MSE (log scale): {mse_log:.4f}")
print(f"Final Test MSE (ms):        {mse_ms:.4f}")
print(f"Final Test RMSE (ms):       {rmse_ms:.1f} ms")


# In[81]:


df_oh = pd.get_dummies(df_cluster["cluster"], prefix="cl")  
#    ergibt (N,100) Matrix mit überall 0/1

# 3.) Kombiniere Deine Features
X_embed  = np.vstack(df_cluster["embedding"].values)      # (N,236)
X_syntax = df_cluster[["word_length","word_pos_in_sentence","log_global_rel"]].values  # (N,3)
X_extra  = df_oh.values                                     # (N,100)

X_comb = np.hstack([X_embed, X_syntax, X_extra])           # (N,339)

df_cluster["log_gaze"] = np.log1p(df_cluster["WORD_GAZE_DURATION"])
y        = torch.from_numpy(df_cluster["log_gaze"].values).float().unsqueeze(1)


X_tensor = torch.from_numpy(X_comb).float()
ds        = TensorDataset(X_tensor, y)
n         = len(ds)
n_train   = int(0.8 * n)
n_temp    = n - n_train
n_val     = n_temp // 2
n_test    = n_temp - n_val

train_ds, val_ds, test_ds = random_split(
    ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# 6) Modell initieren mit input_dim=239
model_PCA_clusters = GazeNet(input_dim=X_comb.shape[1]).to(device)


# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_PCA_clusters.parameters(), lr=1e-3)

# --- Training Loop mit Validation ---
n_epochs = 50
train_mses = []
val_mses   = []

for epoch in range(1, n_epochs+1):
    # --- Training ---
    model_PCA_clusters.train()
    total_train = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model_PCA_clusters(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_train += loss.item() * Xb.size(0)
    train_mse = total_train / len(train_loader.dataset)
    train_mses.append(train_mse)

    # --- Validation auf val_loader ---
    model_PCA_clusters.eval()
    total_val = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model_PCA_clusters(Xb)
            total_val += criterion(preds, yb).item() * Xb.size(0)
    val_mse = total_val / len(val_loader.dataset)
    val_mses.append(val_mse)

    print(f"Epoch {epoch:02d}/{n_epochs} — Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")

# --- Plot Training vs. Validation MSE ---
import matplotlib.pyplot as plt

epochs = list(range(1, n_epochs+1))
plt.figure(figsize=(8,5))
plt.plot(epochs, train_mses, label="Train MSE", marker='o')
plt.plot(epochs, val_mses,   label="Val   MSE", marker='s')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Combined Model on PCA-Reduced Data (w/ Cluster IDs) — Train vs. Validation MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[82]:


# Modell in Eval-Mode
model_PCA_clusters.eval()
preds_log, trues_log = [], []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out = model_PCA_clusters(Xb)
        preds_log.append(out.cpu().numpy())
        trues_log.append(yb.cpu().numpy())

# Arrays zusammenführen
preds_log = np.vstack(preds_log).ravel()
trues_log = np.vstack(trues_log).ravel()

# Log-MSE
mse_log = np.mean((preds_log - trues_log)**2)

# Rücktransform und Original-MSE/RMSE
preds_ms = np.expm1(preds_log)
trues_ms = np.expm1(trues_log)
mse_ms  = np.mean((preds_ms - trues_ms)**2)
rmse_ms = np.sqrt(mse_ms)

print(f"Final Test MSE (log scale): {mse_log:.4f}")
print(f"Final Test MSE (ms):        {mse_ms:.4f}")
print(f"Final Test RMSE (ms):       {rmse_ms:.1f} ms")

