```markdown
# 🧬 Gene Causal AI: Explainable Gene Prioritization Pipeline

A lightweight AI pipeline that integrates literature mining, embeddings, and uncertainty-aware causal gene prediction using open biomedical models such as BioGPT and sentence-transformers.
---

## 📘 Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Output](#example-output)
- [Methodology](#methodology)
- [Uncertainty Interpretation](#uncertainty-interpretation)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

---

## 🧭 Overview

This project improves the **accuracy and explainability of causal gene prediction** by integrating:
- AI-driven literature mining from PubMed and bioRxiv,
- Gene-level embedding and similarity search using FAISS,
- Prediction confidence estimation with Monte Carlo dropout.

The pipeline identifies potential causal genes for complex phenotypes (e.g., Alzheimer’s) based on literature evidence and model confidence.



## ⚙️ Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/ibtj21/Integrating_AI_and_Scientific_Literature_for_Explainable_Causal_Gene_Prediction_iCog-Labs
````

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your PubMed API key :**

   ```bash
   export API_KEY=your_pubmed_api_key
   ```

---

## 🚀 Usage

Run the full pipeline with:

```bash
python main.py
```

You can modify the search query in `main.py` to analyze a different phenotype or gene set.

---

## 🧩 Example Output

```
🎉 Final Causal Gene Prediction:
Predicted gene: CD33
Confidence score: 0.57

Explanation:
ACCORDING TO THE RETRIEVED DOCUMENTS, **CD33** is linked to Alzheimer pathology through lipid droplet regulation in microglia.

Method note: This prediction used 5 Monte Carlo dropout passes to estimate model uncertainty.
```

---

## 🧪 Methodology

1. **Literature Retrieval**
   Retrieve PubMed abstracts for candidate genes related to a phenotype query.

2. **Embedding & Similarity Search**
   Use MiniLM embeddings and FAISS to find the most relevant documents supporting each candidate gene.

3. **Prediction (BioGPT)**
   Rank genes based on semantic relevance and literature context using a causal language model.

4. **Uncertainty Estimation (Monte Carlo Dropout)**
   Estimate prediction confidence and model stability by averaging multiple dropout-enabled forward passes.

5. **Explainability**
   Extract supporting sentences from retrieved literature that mention the top predicted gene to justify predictions.

---

## 📊 Uncertainty Interpretation

Monte Carlo Dropout produces a variability score for the top gene. This **uncertainty** reflects how stable the model’s prediction is across runs.

| Uncertainty (std) | Confidence Level | Interpretation                                          |
| :---------------: | ---------------- | ------------------------------------------------------- |
|     0.0 – 0.3     | 🔒 High          | Strong model agreement — confident prediction           |
|     0.3 – 0.8     | ⚖️ Moderate      | Some variability; moderate confidence                   |
|     0.8 – 1.5     | ⚠️ Low           | Model is unsure; interpret with caution                 |
|        >1.5       | 🚨 Very Low      | Model highly uncertain; multiple genes may be plausible |

Example:

> **Confidence = 0.57, Uncertainty = 1.28 → moderate confidence, high uncertainty**
> CD33 is favored, but other Alzheimer genes like PICALM or CR1 might also be mechanistically relevant.

---

## 🙏 Acknowledgments

* **BioGPT** by Microsoft Research
* **PubMed API** for biomedical literature access
* **Sentence-transformers** and **FAISS** for semantic search

---

> 🧠 *“Explainability is not just about what the model predicts — it’s about showing why it believes it.”*

```

