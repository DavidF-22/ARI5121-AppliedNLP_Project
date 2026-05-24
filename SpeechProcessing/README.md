# 💬 ARI5121 - NLP Speech Processing Speaker Verification Analysis

This folder contains the speech processing component of the ARI5121 Applied NLP project.  
The project evaluates speaker similarity using the **ABI-1 Corpus** and a pre-trained WavLM speaker verification model.

The main notebook implements a pipeline that:

- extracts the required `shortpassage` WAV files from the ABI-1 Corpus;
- creates a metadata CSV containing accent, gender, speaker ID, and file paths;
- extracts speaker embeddings using `microsoft/wavlm-base-plus-sv`;
- compares speaker embeddings using cosine similarity;
- evaluates same-speaker and different-speaker predictions using threshold testing;
- produces evaluation metrics and visualisations, including confusion matrices, PCA, and t-SNE plots.

## 🗃️ Folder Structure

```bash
SpeechProcessing/
│
├── ABI-1 Corpus/   # Dataset folder - not included in this repository
│
├── Docs/
│   ├── ARI5121_AppliedNLP_SpeechProcessing_Report.pdf
│   └── Assignment2026.pdf
│
├── src/
│   └── SVS_Analysis_and_Eval.ipynb
│
├── extracted_wavs/ # Generated after running the notebook
├── results/        # Generated after running the notebook
│
└── README.md
```

## 📂 Important Dataset Placement

Before running the notebook, the ABI-1 Corpus dataset must be placed in the root of this folder:
```text
SpeechProcessing/ABI-1 Corpus/
```

This is required because the notebook uses the following relative path:
```python
CORPUS_DIR = Path("../ABI-1 Corpus")
```

## ⚙️ Setup

Create and activate a virtual environment from inside the SpeechProcessing folder.

### Linux / WSL / macOS

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
```

### Windows PowerShell

```bash
$ python -m venv .venv
$ venv/Scripts/activate
```

Then install the required packages:

```bash
$ pip install -r requirements.txt

# OR

$ pip install ipykernel numpy pandas librosa torch transformers scikit-learn matplotlib
```

## 🗒️ Notes

The ABI-1 Corpus is not included in this repository and must be added manually before running the notebook.

Some processing stages may take time, especially embedding extraction and pairwise similarity computation. The notebook includes checks that reuse existing output files when available, so repeated runs do not always recompute every stage.

The notebook automatically uses CUDA if a compatible GPU is available. Otherwise, it runs on CPU.

## 📧 Contact

For any inquiries or feedback, please contact [David Farrugia](mailto:david.farrugia.22@um.edu.mt).