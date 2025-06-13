#!/usr/bin/env python
# coding: utf-8

# ___
# ### Import data, all needed packages and display data

# In[2]:


import pandas as pd
import numpy as np
import re
import nltk
nltk.download("brown")
from nltk.corpus import brown
from collections import Counter
pd.set_option("display.max_rows", 100)
pd.set_option("display.expand_frame_repr", False)


# In[3]:


use_cols = ["WORD", "WORD_GAZE_DURATION"]
df = pd.read_csv("MonolingualReadingData.csv", usecols=use_cols)
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


print(df[["is_sentence_start", "is_sentence_end"]].head(40))


# In[6]:


print(df[["WORD", "is_sentence_end", "sentence_id", "word_pos_in_sentence"]].tail(80))


# In[7]:


for i in range(13):
    print(df[(df["sentence_id"] == 3) & (df["word_pos_in_sentence"] == i)]["WORD"])


# In[8]:


words = df[(df["sentence_id"] == 3)]["WORD"].tolist()
print(words)


# ___
# ### Import text from experiment and nltk corpus for global and local frequency calculation

# In[9]:


with open("Corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text[:500])
text_tokens = re.findall(r"\b\w+\b", text.lower())
print(len(text_tokens))


# In[10]:


brown_tokens = [w.lower() for w in brown.words() if w.isalpha()]
print(brown_tokens[:50])
print(len(brown_tokens))


# In[11]:


brown_counter = Counter(brown_tokens)
text_counter = Counter(text_tokens)
total_brown = sum(brown_counter.values())
total_text = sum(text_counter.values())


# In[12]:


print(brown_counter)


# ___
# ### Calculation of global and local frequencies

# In[61]:


df["word_lower"] = df["WORD"].str.replace(r"[^a-zA-Z]", "", regex=True).str.lower()
df["global_freq_abs"] = df["word_lower"].map(brown_counter).fillna(0)
df["local_freq_abs"] = df["word_lower"].map(text_counter).fillna(0)
df["global_freq_rel"] = df["global_freq_abs"]/total_brown
df["local_freq_rel"] = df["local_freq_abs"]/total_text
df["log_global_rel"] = np.log1p(df["global_freq_rel"])
df["log_local_rel"] = np.log1p(df["local_freq_rel"])
print(df[["WORD", "word_lower", "global_freq_abs", "local_freq_abs", "global_freq_rel", "local_freq_rel", "log_global_rel", "log_local_rel"]].tail(80))


# In[62]:


df[["log_global_rel", "log_local_rel"]].corr()


# ___
# ### Adding word length column

# In[14]:


df["word_length"] = df["word_lower"].str.len()
print(df[["word_lower", "word_length", "word_pos_in_sentence", "log_global_rel", "log_local_rel", "WORD_GAZE_DURATION"]].tail(80))


# ___
# ### Linear model using grid search(unvectorized and the vectorized for better performance)

# In[56]:


df_clean = df[df["WORD_GAZE_DURATION"].apply(lambda x: str(x).isdigit())].copy()

df_clean["WORD_GAZE_DURATION"] = df_clean["WORD_GAZE_DURATION"].astype(float)

print(df_clean[["word_lower", "word_length", "word_pos_in_sentence", "log_global_rel", "log_local_rel", "WORD_GAZE_DURATION"]].tail(80))


# In[33]:


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


weights = [1,1,1]
bias = 1

best_weights, best_mse, _ = grid_search(5, 0.5, weights, bias)
print(best_mse)
print(best_weights)


# In[46]:


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


init_weights = np.random.uniform(-10, 10, size=3).tolist()
init_bias = np.random.uniform(-10, 10)

best_weights, best_bias, best_mse, results = grid_search(X, y, param_range=20, stepsize=2, init_weights=init_weights, init_bias=init_bias)

print("Best Weights:", best_weights)
print("Best bias:", best_bias)
print("Best MSE:", best_mse)


# In[47]:


init_weights = best_weights
bias = best_bias
best_weights, best_bias, best_mse, results = grid_search(X, y, param_range=5, stepsize=0.5, init_weights=init_weights, init_bias=bias)

print("Best Weights:", best_weights)
print("Best bias:", best_bias)
print("Best MSE:", best_mse)


# In[48]:


init_weights = best_weights
bias = best_bias
best_weights, best_bias, best_mse, results = grid_search(X, y, param_range=2, stepsize=0.2, init_weights=init_weights, init_bias=bias)

print("Best Weights:", best_weights)
print("Best bias:", best_bias)
print("Best MSE:", best_mse)


# ___
# ### Comparison with Gradient Descent-Based Regression Model

# In[79]:


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


# In[81]:


import matplotlib.pyplot as plt

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

