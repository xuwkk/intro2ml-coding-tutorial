# Machine Learning Tutorials — MECH70039

Code for the **Introduction to Machine Learning** tutorials of **MECH70039: Data Science and Digitalisation in the Energy Sector**, MSc in Sustainable Energy Futures, Energy Future Lab, Imperial College London.

Course content (syllabus and lecture slides) can be found [here](https://xuwkk.github.io/wangkun_xu/teaching_datascience.html).

---

## Contents

| Script | Description |
|--------|-------------|
| **`svm.py`** | Binary classification with SVM on the `make_moons` dataset (non-linearly separable). |
| **`pca.py`** | Dimension reduction and 2D visualization with PCA on the Wine dataset. |
| **`kmeans+pca.py`** | K-Means clustering on the Iris dataset, with 2D visualization via PCA. |
| **`chronos_example.py`** | Energy price forecasting with [Chronos](https://github.com/amazon-science/chronos-forecasting) (zero-shot time-series). |

---

## Getting Started

### Dependencies

**Core (all tutorials):**

- `numpy`
- `matplotlib`
- `scikit-learn`

**Chronos only** (for `chronos_example.py`):

- `pandas[pyarrow]`
- `chronos-forecasting`

The easiest way to run the tutorials is **[Google Colab](https://colab.research.google.com/)** so you don’t need to install dependencies locally. Start by installing the dependencies in the interactive window:
```python
%pip install chronos-forecasting
```

Chronos can use a GPU if CUDA is available; otherwise it runs on CPU.
For the `chronos_example.py` tutorial, you can switch between CPU and GPU by clicking the "Runtime" menu and selecting "Change runtime type". 

To run locally, use a virtual environment and install the packages below.

```bash
pip install numpy matplotlib scikit-learn "pandas[pyarrow]" chronos-forecasting
```

---

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.

- **You may**: use, copy, and modify the code for non-commercial purposes (e.g. teaching, learning, research).
- **You must**: give appropriate credit (attribution).
- **You may not**: use the code for commercial purposes without separate permission from the author.
