# %%
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# %%
# Download the full zip but extract only 3 languages
!kaggle datasets download -d crowdai/indian-language-speech-dataset -p /content --unzip

# %%
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# %% [markdown]
# PartA

# %%
data_path = '/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset'  # Change if your folder name differs
languages = os.listdir(data_path)
print("Languages:", languages)

# %%
def extract_mfcc_features(file_path, n_mfcc=13, max_pad_len=174):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Padding for uniform shape
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
        
    return mfcc


# %%
sample_languages = ['Hindi', 'Tamil', 'Bengali']  # Change as needed
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

for idx, lang in enumerate(sample_languages):
    lang_path = os.path.join(data_path, lang)
    sample_file = os.listdir(lang_path)[0]
    mfcc = extract_mfcc_features(os.path.join(lang_path, sample_file))
    
    librosa.display.specshow(mfcc, ax=axs[idx], x_axis='time')
    axs[idx].set_title(f'MFCC - {lang.capitalize()}')
    axs[idx].set_ylabel('MFCC Coefficients')
    axs[idx].set_xlabel('Time')

plt.tight_layout()
plt.show()


# %%
def language_statistics(lang):
    lang_path = os.path.join(data_path, lang)
    mfccs = []
    
    for file in os.listdir(lang_path)[:10]:  # Analyze only 10 samples for speed
        mfcc = extract_mfcc_features(os.path.join(lang_path, file))
        mfccs.append(mfcc.mean(axis=1))  # Mean over time axis

    mfccs = np.array(mfccs)
    return mfccs.mean(axis=0), mfccs.std(axis=0)

for lang in sample_languages:
    mean, std = language_statistics(lang)
    print(f"{lang.capitalize()} - Mean MFCC:", mean.round(2))
    print(f"{lang.capitalize()} - Std MFCC:", std.round(2))


# %% [markdown]
# PartB

# %%
features = []
labels = []

for lang in sample_languages:
    lang_path = os.path.join(data_path, lang)
    for file in os.listdir(lang_path):
        mfcc = extract_mfcc_features(os.path.join(lang_path, file))
        features.append(mfcc.mean(axis=1))  # Use mean of MFCCs as feature
        labels.append(lang)

X = np.array(features)
y = np.array(labels)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
clf = SVC(kernel='rbf', C=1,)
clf.fit(X_train_scaled, y_train)

# %%
y_pred = clf.predict(X_test_scaled)

print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# %%



