#!/usr/bin/env python
# coding: utf-8

# ___
# ### Import data, all needed packages and display data

# In[16]:


import pandas as pd # type: ignore
import numpy as np # type: ignore
import re
import nltk # type: ignore
nltk.download("brown")
from nltk.corpus import brown # type: ignore
from collections import Counter
pd.set_option("display.max_rows", 100)
pd.set_option("display.expand_frame_repr", False)

from dotenv import load_dotenv # type: ignore
import os
import openai # type: ignore
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# In[17]:


import torch # type: ignore
print(torch.cuda.is_available())           # → True
print(torch.cuda.get_device_name(0))       # → NVIDIA GeForce RTX 4060 Ti


# In[18]:


use_cols = ["WORD", "WORD_GAZE_DURATION"]
df = pd.read_csv("Data\MonolingualReadingData.csv", usecols=use_cols)
print(df.head())


# ___
# ### Create new columns that are needed for later modeling

# In[19]:


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


# In[20]:


print(df[["is_sentence_start", "is_sentence_end"]].head(20))


# In[21]:


print(df[["WORD", "is_sentence_end", "sentence_id", "word_pos_in_sentence"]].tail(20))


# ___
# ### Import text from experiment and nltk corpus for global and local frequency calculation

# In[22]:


with open("Data\Corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:500])


# In[23]:


brown_tokens = [w.lower() for w in brown.words() if w.isalpha()]
text_tokens = re.findall(r"\b\w+\b", text.lower())


# In[24]:


brown_counter = Counter(brown_tokens)
text_counter = Counter(text_tokens)
total_brown = sum(brown_counter.values())
total_text = sum(text_counter.values())

print(brown_counter)
print(text_counter)


# ___
# ### Calculation of global and local frequencies

# In[25]:


df["word_lower"] = df["WORD"].str.replace(r"[^a-zA-Z]", "", regex=True).str.lower()
df["global_freq_abs"] = df["word_lower"].map(brown_counter).fillna(0)
df["local_freq_abs"] = df["word_lower"].map(text_counter).fillna(0)
df["global_freq_rel"] = df["global_freq_abs"]/total_brown
df["local_freq_rel"] = df["local_freq_abs"]/total_text
df["log_global_rel"] = np.log1p(df["global_freq_rel"])
df["log_local_rel"] = np.log1p(df["local_freq_rel"])
print(df[["WORD", "word_lower", "global_freq_abs", "local_freq_abs", "global_freq_rel", "local_freq_rel", "log_global_rel", "log_local_rel"]].tail(20))


# In[26]:


df[["log_global_rel", "log_local_rel"]].corr()


# ___
# ### Adding word length column

# In[27]:


df["word_length"] = df["word_lower"].str.len()
print(df[["word_lower", "word_length", "word_pos_in_sentence", "log_global_rel", "log_local_rel", "WORD_GAZE_DURATION"]].tail(20))


# ___
# ### Linear model using grid search (unvectorized and then vectorized for better performance)

# In[28]:


df_clean = df[df["WORD_GAZE_DURATION"].apply(lambda x: str(x).isdigit())].copy()
df_clean["WORD_GAZE_DURATION"] = df_clean["WORD_GAZE_DURATION"].astype(float)


# In[29]:


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


# In[ ]:


init_weights = np.random.uniform(-10, 10, size=3).tolist()
init_bias = np.random.uniform(-10, 10)

best_weights, best_mse, _ = grid_search(2, 0.5, init_weights, init_bias)

print("Best Weights:", best_weights)
print("Best MSE:", best_mse)


# > **Performance Warning**  
# This pure-Python grid search loops over every row for each weight combo—extremely slow. We’ll switch to a fully vectorized approach instead.
# ___

# In[39]:


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


# In[42]:


# Settings: (param_range, stepsize)
refinements = [(20, 2), (5, 0.5), (2, 0.2)]

# Initialisation
init_weights = np.random.uniform(-10, 10, size=3).tolist()
init_bias    = np.random.uniform(-10, 10)

for pr, ss in refinements:
    best_weights, best_bias, best_mse, _ = grid_search(
        X, y,
        param_range=pr,
        stepsize=ss,
        init_weights=init_weights,
        init_bias=init_bias
    )
    print(f"Range={pr}, Step={ss} → Best MSE: {best_mse:.2f}, Weights: {best_weights}, Bias: {best_bias}")
    # for the next refinement run
    init_weights = best_weights
    init_bias    = best_bias


# > **Performance Warning**  
# This is still extremely slow. We’ll switch to a fully gradient descent-based approach instead.

# ___
# ### Comparison with Gradient Descent-Based Regression Model

# In[34]:


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


# In[35]:


import matplotlib.pyplot as plt # type: ignore

# mse_list kommt direkt aus deinem gradient_descent()-Output
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

print(comparison_df.head(10))


# ___
# #### The following cell fetches GPT embeddings for all unique words and saves them to a local cache file (`embeddings_dict.pkl`). On later runs, there is <span style="color:red">no need to run the following cell</span>.  
# 

# In[ ]:


from openai import OpenAI # type: ignore
import time
import openai # type: ignore
import pickle

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


# ___
# ### Loading the pkl file that contains the embeddings and checking for errors

# In[ ]:


import torch # type: ignore
from torch.utils.data import TensorDataset, random_split, DataLoader # type: ignore
import pickle

unique_words = df_clean["word_lower"].dropna().unique()

#Load and check data
with open("embeddings_dict.pkl", "rb") as f:
    embedding_dict = pickle.load(f)


print(f"Entries: {len(embedding_dict)}")


df_emb = pd.DataFrame({
    "word": list(embedding_dict.keys()),
    "emb_length": [len(v) for v in embedding_dict.values()],
    "sample_vals": [v[:5] for v in embedding_dict.values()]
})

print(df_emb.head())

missing = set(unique_words) - set(embedding_dict.keys())
print(f"Words without embedding: {len(missing)}")


# In[ ]:


df_clean["embedding"] = df_clean["word_lower"].map(embedding_dict)
print(len(df_clean))
df_clean = df_clean.dropna(subset=["embedding"]).reset_index(drop=True)

X_embed = np.vstack(df_clean["embedding"].values)
y       = df_clean["WORD_GAZE_DURATION"].values  


# In[ ]:


print("Samples in X:", X_embed.shape[0])
print("Samples in y:", y.shape[0])
assert X_embed.shape[0] == y.shape[0], "Längen stimmen nicht überein!"


# In[ ]:


import torch # type: ignore
from torch.utils.data import TensorDataset, random_split, DataLoader # type: ignore

df_clean["log_gaze"] = np.log1p(df_clean["WORD_GAZE_DURATION"])

y = torch.from_numpy(df_clean["log_gaze"].values).float().unsqueeze(1)

X = torch.from_numpy(X_embed).float()
ds = TensorDataset(X, y)

n_train = int(len(ds) * 0.8)
n_test  = len(ds) - n_train
train_ds, test_ds = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64)


# In[ ]:


import torch.nn as nn # type: ignore
from torch.utils.data import TensorDataset, random_split, DataLoader # type: ignore
from torch.utils.tensorboard import SummaryWriter # type: ignore


# In[ ]:


# ─── Modell auf GPU ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class GazeNet(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

model = GazeNet(emb_dim=X.shape[1]).to(device)

# ─── Loss, Optimizer, Scheduler & TensorBoard ────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
writer = SummaryWriter("runs/gaze_duration_experiment_v2")

# ─── Trainingsloop ────────────────────────────────────────────────────────────
n_epochs = 50
global_step = 0

for epoch in range(1, n_epochs + 1):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for batch_idx, (Xb, yb) in enumerate(train_loader):
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * Xb.size(0)
        # Batch-Logging
        writer.add_scalar("Loss/Train_batch", loss.item(), global_step)
        writer.flush()
        global_step += 1

    train_mse = running_loss / len(train_loader.dataset)
    writer.add_scalar("MSE/Train_epoch", train_mse, epoch)
    writer.flush()

    # --- Histogram logging einmal pro Epoche ---
    for name, param in model.named_parameters():
        writer.add_histogram(f"Weights/{name}", param, epoch)
        writer.flush()

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            preds = model(Xb)
            val_loss += criterion(preds, yb).item() * Xb.size(0)

    val_mse = val_loss / len(test_loader.dataset)
    writer.add_scalar("MSE/Validation", val_mse, epoch)
    writer.flush()

    print(f"Epoch {epoch:02d}/{n_epochs} — Train MSE: {train_mse:.2f}, Val MSE: {val_mse:.2f}")
    scheduler.step()

# ─── Abschluss ────────────────────────────────────────────────────────────────
writer.close()

# ─── TensorBoard starten im Terminal ───────────────────────────────────────────
# tensorboard --logdir=runs/gaze_duration_experiment_v2 --host=0.0.0.0 --port=6006
# Im Browser: http://localhost:6006

