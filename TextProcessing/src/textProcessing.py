# %% [markdown]
# # ARI5121 – Applied NLP | Text Classification Replication Study

# %% [markdown]
# **Student:** David Farrugia <br>
# **Paper:** [NLP based text classification using TF-IDF enabled fine-tuned long short-term memory: An empirical analysis](https://doi.org/10.1016/j.array.2025.100467)  
# **Dataset:** [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
# **Task:** Multi-class news source classification (6 categories)
# 
# ---
# 
# This notebook performs a **replication study** of the fine-tuned CNN-LSTM model described in the selected paper. The primary replication target is the paper’s reported **validation accuracy of approximately 81% over 50 epochs**.
# 
# To achieve this, this jupyter notebook covers the full pipeline:
# > **preprocessing → feature extraction → baseline training → CNN-LSTM training → evaluation → inference**
# 
# **Models trained (Replication):**
# - 5 classical ML baselines (Decision Tree, Naïve Bayes, Linear SVM, KNN, Random Forest)
# - 1 DL baseline - BiLSTM
# - 3 CNN-LSTM models _(3 vocabulary sizes: 10k, ~102k, 200k - 1 model per vocabulary size)_
# 
# **Models trained (Post-Replication):**
# - 3 CNN-LSTM + Multi-Head Attention models _(way-forward extension, same 3 vocab sizes)_
# 
# ---

# %% [markdown]
# ## 0. Setup

# %% [markdown]
# Import all required libraries and fix the global random seed (`SEED = 42`) for reproducibility across NumPy, Python's `random` module, and TensorFlow.
# 
# **Key libraries:**
# - `tensorflow / keras` — deep learning (CNN-LSTM, Attention)
# - `scikit-learn` — ML baselines, TF-IDF, metrics
# - `nltk` — English stopword list
# - `plotly / kaleido` — interactive plots and static export
# 

# %%
# imports
import re
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
 
import nltk
from nltk.corpus import stopwords
 
import tensorflow as tf
from keras import Model
import keras.layers as layers
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import TextVectorization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, 
                             accuracy_score, 
                             roc_auc_score, 
                             roc_curve,
                             precision_recall_curve, 
                             average_precision_score)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import kaleido
kaleido.get_chrome_sync()

nltk.download('stopwords')
nltk.download('punkt')

# initialise random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %%
start_time = time.time()

# %% [markdown]
# ## 1. Loading Dataset

# %% [markdown]
# Before loading the dataset, a validation step is performed to ensure that the required files are correctly located and accessible. The two CSV files are defined using a structured directory approach, which improves portability and reproducibility across different environments.
# 
# A custom validation function is implemented to check three conditions for each dataset file: whether the file exists, whether it is a valid file (and not a directory), and whether it has the correct `.csv` format. If any of these checks fail, a descriptive error is raised, guiding the user to download the dataset from the official Kaggle source. Otherwise, a confirmation message is printed to indicate successful validation.
# 
# Both datasets are read using Pandas and concatenated into a single dataframe, forming a unified dataset/corpus for multi-class classification. We then perform data cleansing to remove rows with missing values in critical fields such as *text* and *subject*, as well as entries containing empty text. These steps are required to avoid negatively affecting model training.
# 
# Lastly, it is worth noting that `when combining Fake.csv and True.csv the unified corpus contian a total of 8 unique category labels`. Therfore, to align with the chosen paper, `dublicate lables are merged` into equivalent classes even though this wasn't explicitly mentioned in the paper. Specifically, _**politicsNews**_ is mapped to _**politics**_ and _**worldnews**_ to _**News**_, resulting in a six multi-class classification problem.

# %%
DATA_DIR = Path("../Kaggle_FakeAndRealNewsDataset")
FAKE_NEWS_CSV = DATA_DIR / "Fake.csv"
TRUE_NEWS_CSV = DATA_DIR / "True.csv"

# validate dataset path
def validate_csv_path(path, label):
    if not path.exists():
        raise FileNotFoundError(f">> [ERROR] {label} not found at: {path.resolve()}\n"
                                f">> [INFO] Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    if not path.is_file():
        raise ValueError(f">> [ERROR] {label} exists but is not a file: {path.resolve()}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f">> [ERROR] {label} is not a CSV file: {path.resolve()}")
    
    print(f">> [OK] {label}: {path.resolve()}")
    
validate_csv_path(FAKE_NEWS_CSV, "Fake news dataset")
validate_csv_path(TRUE_NEWS_CSV, "True news dataset")

# %%
df_false = pd.read_csv(FAKE_NEWS_CSV)
df_true = pd.read_csv(TRUE_NEWS_CSV)

# Combine datasets
df = pd.concat([df_false, df_true], ignore_index=True)

# Drop rows with missing text or subject
df = df.dropna(subset=["text", "subject"])
df = df[df["text"].str.strip() != ""]

# Combine pliticalNews with politics and worldnews with news to achieve the reported 6 cathegories
df['subject'] = df['subject'].replace({
    'politicsNews': 'politics', 
    'worldnews': 'News'
    }) 

print(f">> Dataset Shape: {df.shape}")
print(f"\n>> --- [Subject Distribution] --- :\n")

display(pd.DataFrame(df['subject'].value_counts()). rename(columns={'subject': 'count'}).reset_index())

# %% [markdown]
# ## 2. Text Preprocessing

# %% [markdown]
# Text preprocessing is applied to transform raw textual data into a structured format suitable for machine learning and deep learning models. The preprocessing stage includes converting all text to lowercase, removing punctuation and non-alphabetic characters, tokenizing the text into individual words, and removing common English stopwords.
# 
# These steps are needed to reduce noise and to ensure that only meaningful words contribute to the model’s learning process. By removing irrelevant tokens and standardising the text format, the models can focus on informative patterns within the data rather than superficial variations.
# 
# The cleaned text is stored in a new column named _**clean_text**_, which serves as the primary input for both traditional machine learning models and deep learning architectures.

# %%
# get stop words
stop_words = set(stopwords.words("english"))

# def function to preprocess text - lowercase, remove punctuation, remove stop words
def preprocess(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    
    # join tokens back to string
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess)

print(">> Preprocessing complete")
display(df.head())

# %% [markdown]
# ## 3. Label Encoding

# %% [markdown]
# Since both machine learning and deep learning models require numerical inputs _(depending on the task at hand)_, the subject labels _(categories)_ are converted into integer values using a label encoding approach.
# 
# This transformation allows the same labels to be used with both classical machine learning algorithms and deep learning models. Moreover, the total number of classes is computed and stored, which is later used to define the output layer of neural networks.
# 
# This stage in the project pipline also ensures a consistent mapping between textual categories and numerical representations, which is improtant when we need to remap them in the evaluation stage.

# %%
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["subject"])

print(f">> --- [Classes] --- :\n{label_encoder.classes_}")
print("\n>> Label mapping:")

for i, cls in enumerate(label_encoder.classes_):
    print(f"  {i}: {cls}")
 
NUM_CLASSES = len(label_encoder.classes_)
print(f"\n>> Number of classes: {NUM_CLASSES}")

# %% [markdown]
# ## 4. Train / Test Split

# %% [markdown]
# The dataset is divided into training and testing subsets using a 70/30 split, following the experimental setup described in the paper. Stratified sampling is applied to preserve the original class distribution in both subsets, ensuring that each category is proportionally represented.
#  
# It is also worth mentioning that during the testing phase, a 70/15/15 Train/Validation/Evaluation split, expecially since the authors of the chosen paper, listed multiple train/test splits due to them having attempted multiple configurations themselves. At the end, the 70/30 split was used, since it was ued for the `Fine-Tuned CNN-LSTM model`, which is the
# 
# It is worth noting that different data splitting strategies were considered during the experimental design. In particular, a 70/15/15 train–validation–test split was initially explored, as the authors of the selected paper reported experimenting with multiple configurations.
# 
# However, for consistency with the primary results reported in the paper, a 70/30 train–test split was used at the end of the day. This choice aligns with the configuration used for the `Fine-Tuned CNN-LSTM model`, which is the main replication target in this study. Maintaining to this split ensures a fair reproduction of the original experimental setup, allowing for a better and more meaningful comparison with the reported results.

# %%
X_text = df["clean_text"].values
y = df["label"].values

# (Paper-aligned) 70-30 split for training and testing, stratified by label
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

print(f">> Train set: {len(X_train_text)} samples ({len(X_train_text)/len(X_text)*100:.2f}%)")
print(f">> Test set:  {len(X_test_text)} samples ({len(X_test_text)/len(X_text)*100:.2f}%)")

# %% [markdown]
# ### 4.1. Computing Class-Weights

# %% [markdown]
# To address class imbalance in the dataset, class weights are computed using a `balanced weighting strategy`. These weights assign higher importance to underrepresented classes during training, making sure that the model does not become biased toward more frequently occuring categories.
# 
# Before the addition of class weights the F1 Scores and ROC-AUC socres reported all ranges from 0.5 to 0.6 with significantly lower Average Precision Scores. With the addition of class weights during training, scores like ROC-AUC now range from 0.8 to 0.9, F1 Scores from 0.7 to 0.8 and Average Precision scores also shows a slight improvement.

# %%
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print(">> --- [Class weights] --- :")
for i, (cls, weight) in enumerate(zip(label_encoder.classes_, class_weights)):
    print(f">> {cls:<20} | {weight:.3f}")

# %% [markdown]
# ## 5. TF-IDF Feature Extraction & Integer sequencing

# %% [markdown]
# To enable machine learning models to process textual data, the cleaned text is transformed into numerical representations using two methods: 
# 1. TF-IDF vectorisation for machine learning models &
# 2. Integer sequence encoding for deep learning models.
# 
# For the five machine learning baselines, the TF-IDF method is applied. TF-IDF converts each document into a sparse numerical vector by weighting words based on how important they are. Words that appear frequently in a specific document but are rare globally receive higher weights, while common words across all documents are down-weighted. This allows the model to focus on terms that contribute more effectively to classification. his helps the model focus on words that are more useful for distinguishing between different classes. The vectoriser is fitted on the training data and then applied to the test data to prevent data leakage and ensure a fair evaluation.

# %% [markdown]
# ### 5.1. TF-IDF for ML baselines

# %%
MAX_FEATURES = 10_000
tfidf_vectoriser = TfidfVectorizer(max_features=MAX_FEATURES)

X_tfidf_train = tfidf_vectoriser.fit_transform(X_train_text)
X_tfidf_test  = tfidf_vectoriser.transform(X_test_text)

print(f">> TF-IDF feature matrix shape (train): {X_tfidf_train.shape}")
print(f">> TF-IDF feature matrix shape (test): {X_tfidf_test.shape}")

# %% [markdown]
# However, deep learning models require sequential numerical inputs rather than sparse vectors. To achieve this, a TextVectorization layer is used to convert text into fixed-length integer sequences. Each word is mapped to an index in a learned vocabulary, and sequences are padded or truncated to a **uniform length of 500** tokens as per the selected paper. Multiple vocabulary sizes _(10,000, 101,637_, and 200,000) are explored to analyse the impact of vocabulary coverage on model performance. The vocabulary size **101,637** was mentioned by the authors of the chosen paper, while the other two were added later on as a potential "way forward". 

# %% [markdown]
# ### 5.2. Integer sequences for CNN-LSTM

# %%
VOCAB_SIZE  = [10_000, 101_637, 200_000]    # 101,637
SEQ_LENGTH  = 500        # Sequence Length 500
EMBED_DIM   = 50         # Embedding Dimension 50

seq_splits = []

for vocab_size in VOCAB_SIZE:
    # Vectorise text on the full training set to build the vocabulary
    vectorise_layer = TextVectorization(
        max_tokens=vocab_size,       # vocabulary cap
        output_mode="int",           # integer indices (same as old texts_to_sequences)
        output_sequence_length=SEQ_LENGTH,  # pads/truncates to fixed length (replaces pad_sequences)
        name=f"text_vectorization_{vocab_size}"  # unique name per vocab size
    )
    vectorise_layer.adapt(X_train_text)
    
    X_seq_train = vectorise_layer(X_train_text).numpy()
    X_seq_test  = vectorise_layer(X_test_text).numpy()
    
    seq_splits.append((
        X_seq_train, X_seq_test,
        y_train, y_test
    ))
    
    print(f">> --- [Vocabulary size: {vocab_size}] --- :")
    print(f">> Train shape: {X_seq_train.shape}")
    print(f">> Test shape:  {X_seq_test.shape}\n")

# %% [markdown]
# This dual representation approach allows for a fair comparison between machine learning models, which rely on statistical feature representations and deep learning models, which learn patterns through embeddings and sequence modelling.

# %% [markdown]
# ## 6. Baselines

# %% [markdown]
# ### 6.1. ML - **Precedent Approaches 1 - 5**

# %% [markdown]
# To establish a performance benchmark, several machine learning models are trained using the TF-IDF feature representation. The selected models include **Decision Tree**, **Naïve Bayes**, **Linear Support Vector Machine (SVM)**, **K-Nearest Neighbours (KNN)**, and **Random Forest**. These models are commonly used in text classification and are also referenced as precedent approaches in the selected paper and they were chosen as they are commonly used in text classification and **they where explicitly mentioned and given their respective sections in the paper while their results where explicitly reported as well.**
# 
# Each model is trained using only the training dataset and then evaluated on the unseen test dataset, then predictions are generated for the test set, and performance is measured using two key metrics: **accuracy** and **weighted F1-score**. Weighted F1-score was chosen as a metric, since it *accounts for class imbalance by combining precision and recall across all classes*.
# 
# The results for each model are displayed in a structured format, allowing for easy comparison across different approaches.

# %%
baseline_models = {
    "Decision Tree  (Prec. 1)": DecisionTreeClassifier(random_state=SEED),
    "Naive Bayes    (Prec. 2)": MultinomialNB(),
    "SVM (Linear)   (Prec. 3)": LinearSVC(random_state=SEED, max_iter=2000),
    "KNN            (Prec. 4)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Random Forest  (Prec. 5)": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
}

baseline_results = {}

print(">> --- [Training ML Baselines] --- :")
 
for name, baseline_model in baseline_models.items():
    print(f">> Training {name}...", end=" ", flush=True)
    
    # Train only on the training split
    baseline_model.fit(X_tfidf_train, y_train)

    # Evaluate on the held-out test split
    predictions = baseline_model.predict(X_tfidf_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = classification_report(y_test, 
                               predictions, 
                               target_names=label_encoder.classes_, 
                               zero_division=0, 
                               output_dict=True)["weighted avg"]["f1-score"]
    
    baseline_results[name] = {"model": baseline_model, "predictions": predictions, "accuracy": accuracy, "f1": f1}
    
    print(f"| Accuracy = {accuracy:.3f}, Weighted F1 = {f1:.3f}")

# %% [markdown]
# ### 6.2. DL - **Precedent Approach 6**

# %% [markdown]
# **BiLSTM Implimentation**
# 
# The paper also explicitly includes a BiLSTM model as a deep learning baseline but does not fully specify its architecture or training configuration.
# 
# To ensure computational feasibility and clarity, this implementation uses a simplified BiLSTM consisting of a single Bidirectional LSTM layer followed by a dense classification head which is the same classification head used for the hybrid CNN-LSTM architecture.
# 
# The model is trained using the same key hyperparameters as the CNN-LSTM model (epochs = 50, batch size = 16, Adam optimizer with gradient clipping), for a fair comparison.
# 
# For consistency with the paper’s primary experimental setup, one BiLSTM baseline is trained, using a vocabulary size of 101,637, which which is taken from the main configuration for the fine-tuned CNN-LSTM model. This avoids unnecessary computation from training multiple BiLSTM variants across different vocabulary sizes.

# %%
def build_bilstm_model(vocab_size, seq_length, embed_dim, lstm_units, num_classes, grad_clip):
    inputs = layers.Input(shape=(seq_length,), name="input_sequence")
    
    # Word embedding layer
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="embedding")(inputs)
    
    # BiLSTM layer
    x = layers.Bidirectional(layers.LSTM(lstm_units, dropout=0.2, return_sequences=False), name="bilstm")(x)
    
    # Fully connected + Softmax
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)
    
    bilstm_model = Model(inputs, outputs, name=f"BiLSTM_Model_Vocab{vocab_size}")
    
    optimiser = Adam(learning_rate=0.001, clipvalue=grad_clip)
    
    # Compile model with sparse_categorical_crossentropy for multi-class classification
    bilstm_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])
    return bilstm_model

# %%
print(f">> Training BiLSTM with vocab size 101,637...", end=" ", flush=True)

# Use the split corresponding to vocab size 101,637 (the second one in the list)
X_seq_train, X_seq_test, y_train_dl, y_test_dl = seq_splits[1]

# Build model for this vocabulary size
bilstm_model = build_bilstm_model(
    vocab_size=101_637,
    seq_length=SEQ_LENGTH,
    embed_dim=EMBED_DIM,
    lstm_units=80,
    num_classes=NUM_CLASSES,
    grad_clip=0.2
)

# Train on the training split (no validation split for paper-aligned execution)
history = bilstm_model.fit(X_seq_train, y_train_dl, epochs=50, batch_size=16, validation_split=0.2, shuffle=True, verbose=1)

# Evaluate on the held-out test split
predictions_prob = bilstm_model.predict(X_seq_test)
predictions = np.argmax(predictions_prob, axis=1)

accuracy = accuracy_score(y_test_dl, predictions)
f1 = classification_report(y_test_dl, predictions, 
                            target_names=label_encoder.classes_,
                            zero_division=0, 
                            output_dict=True)["weighted avg"]["f1-score"]

print(f"\n>> BiLSTM (Vocab 101,637) | Accuracy = {accuracy:.3f}, Weighted F1 = {f1:.3f}")

# %% [markdown]
# ## 7. CNN-LSTM Model

# %% [markdown]
# ### 7.1. Building CNN-LSTM Models

# %% [markdown]
# This section defines the `CNN-LSTM` architecture used albeit modified to meet the requirements of this project. The model combines an **embedding layer**, a **one-dimensional convolutional** layer, a **max-pooling** layer, an **LSTM** layer, and a **fully connected softmax output layers**. The `embedding layer` converts integer-encoded words into dense vector representations, allowing the model to learn information from the text. The `convolutional layer` is then used to detect important patterns, such as short word combinations or informative phrases, while the `max-pooling layer` reduces the sequence length and keeps the most important features. After this, the `LSTM layer` is used to learn sequential relationships across the text before the `final fully connected layers` perform the multi-class classification.
# 
# **Modifications:**<br>
# The function is designed to build two related model variants. The first is the main CNN-LSTM model used for replication. The second is a modified version used for the "way forward" extension of this project. In the modified version, **L2 regularisation** is applied to the embedding layer, the **LSTM returns full sequences**, and a **Multi-Head Attention layer** is added before **global average pooling**. This allows the extended model to place more focus on important parts of the sequence before classification. A **dropout layer** is also applied after the attention output to help reduce overfitting.

# %%
def build_cnn_lstm_modified(vocab_size, seq_length, embed_dim, lstm_units, num_classes, learning_rate, grad_clip, has_modifications=False):
    """
    Builds the Fine-Tuned CNN-LSTM model.
 
    Architecture:
        - Embedding → Conv1D → MaxPooling1D → LSTM → Dense → Softmax
        
    Parameters:
        - vocab_size: Size of the vocabulary (input_dim for Embedding).
        - seq_length: Fixed input sequence length (e.g., 500).
        - embed_dim: Dimension of the word embeddings (output_dim for Embedding).
        - lstm_units: Number of hidden units in the LSTM layer.
        - num_classes: Number of output classes for the final Dense layer.
        - learning_rate: Learning rate for the Adam optimizer.
        - grad_clip: Gradient clipping value for the Adam Optimiser.
        - has_modifications: Whether to include modifications to CNN-LSTM architecture.
        
    Note: input_length is omitted from Embedding because TextVectorization
    already guarantees fixed-length sequences (SEQ_LENGTH=500), so Keras
    can work out the shape automatically.
    """
    
    name = f"CNN_LSTM_Model_Vocab{vocab_size}"
    inputs = layers.Input(shape=(seq_length,), name="input_sequence")
    
    # Word embedding layer
    if has_modifications:
        x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_regularizer=l2(1e-4), name="embedding")(inputs)
    else:
        x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="embedding")(inputs)
    
    # Convolutional layer
    x = layers.Conv1D(filters=64, kernel_size=5, activation="relu", name="conv1d")(x)
    # Max-pooling layer
    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)
    
    if has_modifications:
        name = f"CNN_LSTM_Modified_Model_Vocab{vocab_size}"
        # return_sequences=True keeps 3D output (batch, steps, features)
        x = layers.LSTM(units=lstm_units, return_sequences=True, name="lstm")(x)
        # Multi-Head Attention
        x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
        # Add dropout after attention to prevent overfitting on the attention outputs
        x = layers.Dropout(0.3, name="attention_dropout")(x)
        # Pool across the sequence dimension before Dense
        x = layers.GlobalAveragePooling1D()(x)
    else:
        # LSTM layer — 80 hidden units
        x = layers.LSTM(units=lstm_units, name="lstm")(x)
    
    # Fully connected layers
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)
    
    cnn_lstm_model = Model(inputs, outputs, name=name)
    
    # Adjust learning rate if modifications are included
    if has_modifications:
        # Attention models are more sensitive to LR — use a lower rate to avoid
        # the loss diverging immediately from epoch 1
        lr = learning_rate * 0.5
    else:
        lr = learning_rate
    
    optimiser = Adam(learning_rate=lr, clipvalue=grad_clip)

    # Compile model with sparse_categorical_crossentropy for multi-class classification
    cnn_lstm_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])
    return cnn_lstm_model

# %% [markdown]
# The models are built using the same general configuration across three vocabulary sizes: _10,000, 101,637, and 200,000_. This allows the effect of vocabulary size on performance to be examined while keeping the rest of the architecture consistent. The `main hyperparameters` include a `sequence length of 500`, an `embedding dimension of 50`, `80 LSTM units`, and an `Adam optimizer with gradient clipping and an initial learning rate of 0.001`.
# 
# It is important to note that the original paper did not provide a complete implementation or a code repository of the architecture. As a result, the model used in this project was reconstructed from reading the methodology section and going through the architecture diagram gradually. To interpret unclear details, external AI tools were used to help with analysing the written methodology and architecture diagram presented in the chosen paper.

# %%
config = {
    "vocab_size": VOCAB_SIZE,
    "seq_length": SEQ_LENGTH,
    "embed_dim": EMBED_DIM,
    "lstm_units": 80,
    "num_classes": NUM_CLASSES,
    "learning_rate": 0.001,
    "grad_clip": 0.2
}

cnn_lstm_models = []
cnn_lstm_modified_models = []

for vocab_size in config["vocab_size"]:
    print(f"\n>> --- [Building CNN-LSTM model with vocab size: {vocab_size:,}] --- :")
    
    cnn_lstm = build_cnn_lstm_modified(
        vocab_size,
        config["seq_length"], 
        config["embed_dim"], 
        config["lstm_units"], 
        config["num_classes"], 
        config["learning_rate"], 
        config["grad_clip"]
    )
    
    cnn_lstm.summary()
    cnn_lstm_models.append(cnn_lstm)
    
    cnn_lstm_modified = build_cnn_lstm_modified(
        vocab_size, 
        config["seq_length"],
        config["embed_dim"], 
        config["lstm_units"], 
        config["num_classes"], 
        config["learning_rate"], 
        config["grad_clip"],
        has_modifications=True
    )

    cnn_lstm_modified.summary()
    cnn_lstm_modified_models.append(cnn_lstm_modified)

# %% [markdown]
# ### 7.2. Callbacks

# %% [markdown]
# This section defines the callbacks used during training to improve learning stability and reduce unnecessary computation. A `ReduceLROnPlateau` callback is included in all training runs. This callback monitors the validation loss and automatically reduces the learning rate when improvement slows down. This helps the optimiser continue learning more effectively when progress begins to plateau.
# 
# Moreover, the `ReduceLROnPlateau` callback was initially included because the origional paper reports experimenting with different learning rates across training runs. However, the exact strategy for adjusting the learning rate **is not explicitly described**. Therefore, this process was automated which provides a more controlled and efficient alternative to manual tuning, while remaining consistent with the paper’s training methodology.
# 
# Additionally, an optional `EarlyStopping` callback is also included. When enabled, this stops training if the validation loss does not improve for a defined number of epochs and restores the best weights observed during training. This is useful for preventing wasted computation and reducing overfitting, especially in longer training runs.

# %%
def get_callbacks(use_early_stopping):
    callbacks = [
        # Reduce LR if val_loss plateaus — helps match paper's manual LR tuning
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    if use_early_stopping:
        # Early stopping to avoid wasted compute
        callbacks.append(EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1))
        
    return callbacks

# %% [markdown]
# ## 8. Training

# %% [markdown]
# This section trains all CNN-LSTM models using the sequence-based training data prepared earlier. Each model is trained for 50 epochs with a batch size of 16, which follows the main training setup described in the paper. During training, 20% of the training data is reserved internally as a validation split. This portion is used to monitor the model’s performance across epochs and to support callbacks such as learning rate reduction and early stopping.
# 
# Two optional training controls are included in this project: `early stopping` and `class weighting`. **Early stopping** is used to prevent unnecessary computation and reduce the risk of overfitting while **class weights** make the learning process more balanced across categories, especially if some classes appear less frequently than others. This helps reduce bias toward majority classes.
# 
# The training process is wrapped inside a reusable function so that the same procedure can be applied to both the standard CNN-LSTM models and the modified CNN-LSTM + Attention models. Each model is trained using the sequence representation that matches its corresponding vocabulary size. The training history for every run is stored so that it can later be analysed and visualised.

# %%
EPOCHS = 50
BATCH_SIZE = 16

use_early_stopping = True
use_class_weights = True

model_histories = []
attention_histories = []

def train_models(model_list, has_modifications=False):
    histories = attention_histories if has_modifications else model_histories

    for model_index, (cnn_lstm, split) in enumerate(zip(model_list, seq_splits)):
        X_seq_train, _, y_train_dl, _ = split

        vocab_size = config["vocab_size"][model_index]
        
        if has_modifications:
            print(f"\n>> Training CNN-LSTM with Modifications and vocab size: {vocab_size:,} ... ", end=" ", flush=True)
        else:
            print(f"\n>> Training CNN-LSTM with vocab size: {vocab_size:,} ... ", end=" ", flush=True)
        
        history = cnn_lstm.fit(
            X_seq_train, y_train_dl,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            shuffle=True,
            callbacks=get_callbacks(use_early_stopping),
            class_weight=class_weight_dict if use_class_weights else None,
            verbose=1
        )
        
        histories.append(history)

train_models(cnn_lstm_models)
train_models(cnn_lstm_modified_models, has_modifications=True)

print("\n>> --- [Training Complete] ---")

# %% [markdown]
# ### 8.1. Saving Models

# %%
wORn_es = "With" if use_early_stopping else "No"
wORn_cw = "With" if use_class_weights else "No"
MODELS_OUTPUT_DIR = Path(f"../Out/Models_{wORn_es}EarlyStopping_{wORn_cw}ClassWeights")
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist

def save_model(cnn_lstm_model_lists, has_modifications=False):
    for i, cnn_lstm in enumerate(cnn_lstm_model_lists):
        if has_modifications:
            model_path = MODELS_OUTPUT_DIR / f"cnn_lstm_modified_model_vocab{config['vocab_size'][i]}.keras"
        else:
            model_path = MODELS_OUTPUT_DIR / f"cnn_lstm_model_vocab{config['vocab_size'][i]}.keras"
        
        # save model to .keras in OUTPUT_DIR
        cnn_lstm.save(model_path)
        if has_modifications:
            print(f">> [OK] Saved CNN-LSTM + Modifications model with vocab size {config['vocab_size'][i]} to: {model_path.resolve()}")
        else:
            print(f">> [OK] Saved CNN-LSTM model with vocab size {config['vocab_size'][i]} to: {model_path.resolve()}")

save_model(cnn_lstm_models)
save_model(cnn_lstm_modified_models, has_modifications=True)

# %% [markdown]
# ### 8.2. Training Curves

# %% [markdown]
# This section visualises the training behaviour of each model by plotting both loss and accuracy across epochs. For every vocabulary size, separate plots are created for the standard CNN-LSTM model and the modified CNN-LSTM + Attention model. In addition to being displayed, the plots are also saved as image files in a dedicated output folder.
# 
# These plots are useful for identifying patterns such as steady learning, overfitting, underfitting, or unstable optimisation. A horizontal reference line is also added at an accuracy of 0.81 to represent the main target reported in the paper. This makes it easier to visually compare the achieved validation accuracy with the paper’s reported result.

# %%
PLOTS_OUTPUT_DIR = MODELS_OUTPUT_DIR / "Training_Plots"
PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist

for history_index, (history, attention_history) in enumerate(zip(model_histories, attention_histories)):
    for hist, model_type in [(history, "CNN-LSTM"), (attention_history, "CNN-LSTM + Attention")]:
        epochs = list(range(1, len(hist.history["loss"]) + 1))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss", "Accuracy")
        )

        # --- Loss curves (left) ---
        fig.add_trace(
            go.Scatter(x=epochs, y=hist.history["loss"],
                    name="Training Loss", line=dict(color="royalblue")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=hist.history["val_loss"],
                    name="Validation Loss", line=dict(color="darkcyan", dash="dash")),
            row=1, col=1
        )

        # --- Accuracy curves (right) ---
        fig.add_trace(
            go.Scatter(x=epochs, y=hist.history["accuracy"],
                    name="Training Accuracy", line=dict(color="darkorange")),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=hist.history["val_accuracy"],
                    name="Validation Accuracy", line=dict(color="darkred", dash="dash")),
            row=1, col=2
        )

        # Paper target accuracy 0.81
        fig.add_hline(
            y=0.81, row=1, col=2,
            line=dict(color="red", dash="dot", width=1.5),
            annotation_text="Paper target (0.81)",
            annotation_position="top left"
        )

        # --- Layout ---
        fig.update_layout(
            title=dict(
                text=f"{model_type} Training Progress (Loss & Accuracy) - Vocabulary Size: {config['vocab_size'][history_index]:,}",
                x=0.5,
                xanchor="center"
            ),
            width=1100, height=450,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            hovermode="x unified"
        )

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss",    row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        
        fig.show()
        
        # Save plot to file
        if model_type == "CNN-LSTM + Attention":
            model_type_str = "cnn_lstm_modified_model"
        else:
            model_type_str = "cnn_lstm_model"
            
        plot_filename = f"{model_type_str}_vocab{config['vocab_size'][history_index]:,}.png"
        fig.write_image(PLOTS_OUTPUT_DIR / plot_filename)

# %% [markdown]
# ## 9. Evaluation

# %% [markdown]
# This section evaluates the trained models on the held-out test set using both numerical metrics and visual performance curves. For each model and vocabulary size, the test sequences are passed through the network to produce predicted class probabilities. These probabilities are then converted into final class predictions by selecting the class with the highest probability score.
# 
# A classification report is generated for every model configuration. This report provides precision, recall, and F1-score for each class, along with overall summary scores. These metrics are useful because they show not only the general performance of the model, but also how well it performs on each individual category.
# 
# For visualisations, Receiver Operating Characteristic Area Under Curve (ROC-AUC) curves and Precision Recall Average Precision (PR-AP) curves are also plotted for each class. Since this is a multi-class classification problem, the labels are converted into a one-vs-rest format so that ROC and PR curves can be calculated for each class separately. The ROC curves show the trade-off between the false positive rate and the true positive rate, while the PR curves show the relationship between precision and recall.

# %%
# colours for the 6 classes — one per class, consistent across all plots
CLASS_COLORS = px.colors.qualitative.Plotly[:NUM_CLASSES]

def plot_roc_curves(y_true, y_pred_probs, class_names, title):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    fig = go.Figure()
    
    # random chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(color="black", dash="dash", width=1),
        name="Random chance",
        showlegend=True
    ))
    
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        auc_score = roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{cls} (AUC = {auc_score:.3f})",
            line=dict(color=CLASS_COLORS[i], width=2)
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=750, height=550,
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
        hovermode="x unified"
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig


def plot_pr_curves(y_true, y_pred_probs, class_names, title):
    """Precision-Recall curve per class — AP score in legend."""
    
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    fig = go.Figure()
    
    for i, cls in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name=f"{cls} (AP = {ap_score:.3f})",
            line=dict(color=CLASS_COLORS[i], width=2)
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=750, height=550,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
        hovermode="x unified"
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig

# %%
EVAL_PLOTS_OUTPUT_DIR = MODELS_OUTPUT_DIR / "Eval_Plots"
EVAL_PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist

def evaluate_model(model_list, sequence_splits, has_modifications=False):
    # --- Run for each vocab size ---
    for model_index, (cnn_lstm, split) in enumerate(zip(model_list, sequence_splits)):
        _, X_seq_test, _, y_test_dl = split

        vocab_label = f"Vocab {config['vocab_size'][model_index]:,}"
        
        # get predicted probabilities
        y_pred_probs = cnn_lstm.predict(X_seq_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # print classification report
        class_report = classification_report(
            y_test_dl, 
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
            digits=3
        )
        print(f">> Classification Report ({vocab_label}):")
        print(class_report)
        
        # --- ROC-AUC ---
        if has_modifications:
            model_type_str = "CNN-LSTM + Modifications"
        else:
            model_type_str = "CNN-LSTM"

        fig_roc = plot_roc_curves(
            y_test_dl, 
            y_pred_probs,
            label_encoder.classes_,
            title=f"{model_type_str} ({vocab_label}) - ROC Curve"
        )
        fig_roc.show()
        # save ROC plot to file
        fig_roc.write_image(EVAL_PLOTS_OUTPUT_DIR / f"{cnn_lstm.name.lower()}_ROC.png")

        # --- PR-AP ---
        fig_pr = plot_pr_curves(
            y_test_dl, y_pred_probs,
            label_encoder.classes_,
            title=f"{model_type_str} ({vocab_label}) - PR Curve"
        )
        
        fig_pr.show()
        # save PR plot to file
        fig_pr.write_image(EVAL_PLOTS_OUTPUT_DIR / f"{cnn_lstm.name.lower()}_PR.png")

print("\n>> --- [Evaluating CNN-LSTM Models] --- :")
evaluate_model(cnn_lstm_models, seq_splits)
print("\n>> --- [Evaluating CNN-LSTM + Modifications Models] --- :")
evaluate_model(cnn_lstm_modified_models, seq_splits, has_modifications=True)

# %% [markdown]
# ### 9.1. Inference Testing

# %% [markdown]
# Lastly, this section performs inference on the test dataset to generate example predictions from each trained model.
# 
# For each model and vocabulary configuration, predictions are generated, then both the predicted labels and their resepctive confidence scores are extracted. A small subset of samples is then selected, consisting of correctly classified examples and incorrectly classified ones. This allows a direct comparison between successful predictions and failure cases.
# 
# For each selected sample, the original text, the cleaned version of the text, the true label, the predicted label, and whether the prediction was correct are recorded. This structured output makes it easier to analyse patterns in model behaviour, such as common types of errors or cases where the model is highly confident but incorrect. The results are combined into a single dataframe and saved as a CSV file.

# %%
def run_inference_on_samples(model_list, seq_splits):
    X_test_text_raw = X_test_text

    all_sample_dfs = []
    
    for (cnn_lstm_model, split) in zip(model_list, seq_splits):
        print(f">> Running Inference on {cnn_lstm_model.name.lower()}")
        _, X_seq_test_raw, _, y_test_sample = split

        y_pred_probs = cnn_lstm_model.predict(X_seq_test_raw, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        correct_idx = np.where(y_pred == y_test_sample)[0][:3]
        incorrect_idx = np.where(y_pred != y_test_sample)[0][:3]
        sample_idx = np.concatenate([correct_idx, incorrect_idx])

        sample_df = pd.DataFrame({
            "model": cnn_lstm_model.name.lower(),
            "true_label": label_encoder.inverse_transform(y_test_sample[sample_idx]),
            "pred_label": label_encoder.inverse_transform(y_pred[sample_idx]),
            "confidence": np.max(y_pred_probs[sample_idx], axis=1),
            "correct": y_pred[sample_idx] == y_test_sample[sample_idx],
            "text": X_test_text_raw[sample_idx],
            "cleaned_text": [preprocess(t) for t in X_test_text_raw[sample_idx]]
        })
        
        all_sample_dfs.append(sample_df)

    return pd.concat(all_sample_dfs, ignore_index=True)

# %%
cnn_lstm_sample_results = run_inference_on_samples(cnn_lstm_models, seq_splits)
attention_sample_results = run_inference_on_samples(cnn_lstm_modified_models, seq_splits)

# combine into one and save to CSV
combined_sample_results = pd.concat([cnn_lstm_sample_results, attention_sample_results],ignore_index=True)

output_file = MODELS_OUTPUT_DIR / f"all_inference_samples_{'with_early_stopping' if use_early_stopping else 'no_early_stopping'}.csv"
combined_sample_results.to_csv(output_file, index=False)

print(f"\n>> [OK] Saved inference samples to: {output_file.resolve()}")

# %% [markdown]
# ## 10. Elapsed Time Tracking

# %% [markdown]
# Time tracking was included in this project to meet the requirements of this project.

# %%
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n>> Total execution time in minutes: {elapsed_time/60:.3f} minutes")

# Add time to a time.txt log file
log_file = MODELS_OUTPUT_DIR / "time.txt"
with open(log_file, "w") as f:
    f.write(f"{elapsed_time/60:.3f} minutes = {elapsed_time/3600:.2f} hours = ~{elapsed_time/3600:.1f}")

# %% [markdown]
# ---
# 
# ## Experiment Logs:
# 
# | Entry | Experiment Name                           | Speed (In Hours) | Key Hyperparemerters | Value                         |
# |:-----:|:-----------------------------------------:|:----------------:|:--------------------:|:-----------------------------:|
# |   1   | Models_NoEarlyStopping_NoClassWeights     |      ~4.5        | Early Stopping <br> Class Weights | False <br> False |
# |   2   | Models_NoEarlyStopping_WithClassWeights   |      ~5.7        | Early Stopping <br> Class Weights | False <br> True  |
# |   3   | Models_WithEarlyStopping_NoClassWeights   |      ~3.5        | Early Stopping <br> Class Weights | True <br> False  |
# |   4   | Models_WithEarlyStopping_WithClassWeights |      ~4.2        | Early Stopping <br> Class Weights | True <br> True   |
# 
# **Consistent Parameters For Each Run:**
# 
# | Parameter             | Value                          |
# |-----------------------|--------------------------------|
# | Vocabulary Sizes      | 10,000; 101,637; 200,000       |
# | Sequence Length       | 500                            |
# | Embedding Dimension   | 50                             |
# | LSTM Units            | 80                             |
# | Number of Classes     | 6                              |
# | Initial Learning Rate | 0.001                          |
# | Gradient Clipping     | 0.2                            |
# | Epochs                | 50                             |
# | Batch Size            | 16                             |


