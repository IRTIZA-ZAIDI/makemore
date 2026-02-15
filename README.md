# makemore

A notebook-first, from-scratch walkthrough for building character-level name generators in PyTorch, following the classic “makemore” progression: start with a simple baseline and gradually add modeling power and training stability techniques.

The dataset is a list of names (`names.txt`). Each notebook builds a model that learns to generate new names character-by-character.

---

## What’s in this repo

### Dataset

* **`names.txt`**
  Training corpus: one name per line.

### Notebooks (recommended order)

1. **`build_makemore.ipynb`**
   Intro / baseline model work. Uses the names dataset to build a character-level modeling pipeline (vocabulary, encoding, training loop, sampling).

2. **`build_makemore02_MLP.ipynb`**
   Moves beyond the baseline to a small neural network (MLP) over a fixed context window (n-gram style context). Introduces embeddings + hidden layer modeling.

3. **`build_makemore02_cleaned.ipynb`**
   A cleaned-up version of the previous notebook (same idea, easier to follow / better organized).

4. **`build_makemore03.ipynb`**
   Training diagnostics and stability work (activation/gradient behavior). Typically where initialization / normalization ideas begin to matter as the model depth increases.

5. **`build_makemore03_cleaned.ipynb`**
   A cleaned-up version focused on stable training as depth increases (the notebook text references keeping activations/gradients well-behaved).

6. **`build_makemore04.ipynb`**
   Exercises / deeper dives (the notebook begins with “Exercise 1”). Often used for practicing implementation details and reinforcing concepts.

7. **`build_makemore05.ipynb`**
   More advanced architecture work (later-stage notebook in the series; typically includes stronger sequence modeling structure compared to the earlier MLP setup).

---

## Project structure

```
.
├── build_makemore.ipynb
├── build_makemore02_MLP.ipynb
├── build_makemore02_cleaned.ipynb
├── build_makemore03.ipynb
├── build_makemore03_cleaned.ipynb
├── build_makemore04.ipynb
├── build_makemore05.ipynb
└── names.txt
```

---

## Requirements

* Python 3.9+ (recommended: 3.10+)
* Jupyter Notebook or JupyterLab
* PyTorch

Common helpful packages (depending on what the notebooks import):

* `numpy`
* `matplotlib`

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install torch jupyter numpy matplotlib
```

---

## Running

Start Jupyter:

```bash
jupyter notebook
```

Then open and run notebooks in order (recommended):

1. `build_makemore.ipynb`
2. `build_makemore02_cleaned.ipynb`
3. `build_makemore03_cleaned.ipynb`
4. `build_makemore04.ipynb`
5. `build_makemore05.ipynb`

Notes:

* The “cleaned” notebooks are usually the better default if you want a clearer narrative.
* All notebooks assume `names.txt` is in the repo root.

---

## Expected outputs

Depending on the notebook:

* training loss curves / printed training loss
* sampled/generated names after training
* diagnostics for activation/gradient distributions (later notebooks)

---

## Tips

* If you run out of memory, reduce batch size or model dimensions inside the notebook cells.
* If training is slow on CPU, consider installing a CUDA-enabled PyTorch build and using an NVIDIA GPU.


