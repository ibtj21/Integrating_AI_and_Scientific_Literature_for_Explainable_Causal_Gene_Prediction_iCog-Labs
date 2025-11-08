# -*- coding: utf-8 -*-
"""
biogpt_gene_ranker.py
Lightweight gene prioritizer using a BioGPT-like causal LM + Monte Carlo Dropout.
"""

import re
import math
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ---------- Config ----------
DEFAULT_MODEL = "microsoft/biogpt"  
DEVICE = "cpu"

# ---------- Utilities ----------
def safe_tokenize(tokenizer, text: str):
    return tokenizer(text, return_tensors="pt", add_special_tokens=True)

def extract_supporting_sentences(top_docs: List[Dict], gene: str, max_sentences: int = 3) -> List[str]:
    """Return sentences from top_docs that mention the gene (case-insensitive)."""
    gene_regex = re.compile(r"\b" + re.escape(gene) + r"\b", flags=re.IGNORECASE)
    found = []
    for d in top_docs:
        content = d.get("content", "")
        # naive sentence split
        sentences = re.split(r'(?<=[.!?])\s+', content)
        for s in sentences:
            if gene_regex.search(s):
                found.append(s.strip())
                if len(found) >= max_sentences:
                    return found
    return found

# ---------- Core class ----------
class BioGPTGeneRanker:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = DEVICE, mc_samples: int = 20):
        self.model_name = model_name
        self.device = device
        self.mc_samples = mc_samples
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        # Keep dropout layers present — we'll toggle model.train() during inference to enable dropout
        # But we still use torch.no_grad() to avoid updating grads.
        # For memory-constrained systems, you may want to use torch.float16 on GPU.
        print(f"Loaded model {self.model_name} on {self.device}")

    def build_prompt(self, phenotype: str, top_docs: List[Dict]) -> str:
        """Construct a concise prompt that includes phenotype and the top documents."""
        prompt_parts = [
            f"Phenotype: {phenotype.strip()}",
            "Relevant literature (top documents):"
        ]
        for i, d in enumerate(top_docs, start=1):
            title = d.get("metadata", {}).get("Title") or d.get("metadata", {}).get("title") or f"Doc{i}"
            content = d.get("content", "")[:800]  # truncate to keep prompt short
            prompt_parts.append(f"[Doc{i}] {title}. {content}")
        prompt_parts.append("\nQuestion: Which of the following genes is most likely causal given the phenotype and these documents?\nAnswer:")
        prompt = "\n".join(prompt_parts)
        return prompt

    def score_candidate_given_prompt(self, prompt: str, candidate: str) -> float:
        """
        Compute negative token log-probability (sum over candidate token sequence) of candidate
        given the prompt using the model. Higher is better (less negative).
        We mask prompt tokens in labels so loss is only computed on candidate tokens.
        """
        # create input tokens for prompt + candidate
        combined = prompt + " " + candidate
        enc_combined = self.tokenizer(combined, return_tensors="pt")
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc_combined["input_ids"].to(self.device)
        attention_mask = enc_combined["attention_mask"].to(self.device)

        prompt_len = enc_prompt["input_ids"].shape[1]
        labels = input_ids.clone()
        # mask prompt tokens so loss counts only candidate tokens
        labels[0, :prompt_len] = -100

        # forward (no_grad)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # outputs.loss is mean loss over unmasked tokens; compute token count to get sum log-prob
            loss = outputs.loss.item()  # averaged NLL
            # number of candidate tokens = total tokens - prompt_len
            total_tok = input_ids.shape[1]
            cand_tok = total_tok - prompt_len
            if cand_tok <= 0:
                # fallback - very unlikely
                return -loss
            # sum negative log-likelihood (NLL_sum = loss * cand_tok)
            nll_sum = loss * cand_tok
            # return negative nll_sum as a score (higher is better)
            return -nll_sum

    def mc_dropout_scores(self, prompt: str, candidates: List[str], T: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Monte Carlo dropout T times. Returns mean_scores (shape [n_candidates])
        and std_scores (shape [n_candidates]).
        """
        T = T or self.mc_samples
        n = len(candidates)
        all_scores = np.zeros((T, n), dtype=float)

        # Ensure dropout active: set model to train() (dropout active), but use no_grad to avoid backprop memory.
        self.model.train()
        for t in range(T):
            # run each candidate scoring under dropout
            scores = []
            for i, c in enumerate(candidates):
                s = self.score_candidate_given_prompt(prompt, c)
                all_scores[t, i] = s
            # Optionally print progress
            if (t + 1) % max(1, T // 5) == 0:
                print(f"MC dropout pass {t+1}/{T} completed.")
        # restore to eval mode
        self.model.eval()
        mean_scores = all_scores.mean(axis=0)
        std_scores = all_scores.std(axis=0)
        return mean_scores, std_scores

    def predict(self, topk_docs: List[Dict], phenotype: str, candidate_genes: List[str], T: int = None) -> Dict:
        """
        Main entrypoint. Returns dict with:
        - predicted_gene
        - mean_probabilities (dict gene->mean prob)
        - mean_scores, std_scores
        - confidence_score (mean prob of predicted gene)
        - explanation (string starting with "ACCORDING TO THE RETRIEVED DOCUMENTS...")
        """
        if not candidate_genes:
            raise ValueError("candidate_genes must be provided and non-empty")
        prompt = self.build_prompt(phenotype, topk_docs)
        T = T or self.mc_samples

        print("🔎 Running Monte Carlo Dropout scoring...")
        mean_scores, std_scores = self.mc_dropout_scores(prompt, candidate_genes, T=T)

        # Convert scores to probabilities via softmax for each MC sample would be ideal,
        # but we only have mean scores aggregated. If we had per-sample we could average probs;
        # here we will softmax the mean_scores to get a final probability distribution.
        # (This is simple and common in ranking setups.)
        exp = np.exp(mean_scores - np.max(mean_scores))
        probs = exp / exp.sum()

        # Choose top gene
        top_idx = int(np.argmax(probs))
        predicted_gene = candidate_genes[top_idx]
        confidence = float(probs[top_idx])
        uncertainty = float(std_scores[top_idx])

        # Build explanation relying on topk_docs:
        explanation_lines = []
        explanation_lines.append("ACCORDING TO THE RETRIEVED DOCUMENTS,")
        # Basic evidence paragraph: mention which docs mention the predicted gene
        supporting_sentences = extract_supporting_sentences(topk_docs, predicted_gene, max_sentences=3)
        if supporting_sentences:
            explanation_lines.append(
                f"the model selects **{predicted_gene}** because it is directly mentioned in the retrieved literature. "
                "Here are supporting excerpts:"
            )
            for s in supporting_sentences:
                explanation_lines.append(f"- \"{s}\"")
        else:
            # fallback rhetorical explanation
            explanation_lines.append(
                f"the model selects **{predicted_gene}** because the combined textual context (phenotype + retrieved abstracts) "
                "contains language and patterns that the model associates with causal gene descriptions for this phenotype."
            )

        # Add short methodological note
        explanation_lines.append(
            f"\nMethod note: the prediction uses {T} Monte Carlo dropout forward passes (dropout kept active at inference) "
            "to approximate model uncertainty. The reported confidence is the normalized probability for the top gene; "
            f"estimated uncertainty (std of scores) for this gene is {uncertainty:.4f} (lower is more certain)."
        )

        explanation = "\n".join(explanation_lines)

        # package results
        result = {
            "predicted_gene": predicted_gene,
            "mean_scores": {g: float(s) for g, s in zip(candidate_genes, mean_scores)},
            "std_scores": {g: float(s) for g, s in zip(candidate_genes, std_scores)},
            "probabilities": {g: float(p) for g, p in zip(candidate_genes, probs)},
            "confidence_score": confidence,
            "explanation": explanation,
            "top_docs_used": topk_docs[:5],
        }
        return result

# ---------- Example usage ----------
if __name__ == "__main__":
    # Example placeholders (replace with your real top_k docs and genes)
    sample_docs = [
        {"content": "Gene ABC1 is implicated in phenotype X by pathway Y. Further evidence comes from ...", "metadata": {"PMID":"111","Title":"ABC1 involvement"}},
        {"content": "Mutations in GENE2 were observed in patients with phenotype X and shown to affect expression of ...", "metadata": {"PMID":"222","Title":"GENE2 mutations"}},
        # add up to top K docs...
    ]
    phenotype = "phenotype X: early-onset neurodegeneration with motor symptoms"
    candidates = ["ABC1", "GENE2", "OTHERGENE"]

    ranker = BioGPTGeneRanker(model_name=DEFAULT_MODEL, device=DEVICE, mc_samples=8)  # reduced MC passes to be lighter
    out = ranker.predict(sample_docs, phenotype, candidates, T=8)
    print("Prediction:", out["predicted_gene"])
    print("Confidence:", out["confidence_score"])
    print(out["explanation"])
