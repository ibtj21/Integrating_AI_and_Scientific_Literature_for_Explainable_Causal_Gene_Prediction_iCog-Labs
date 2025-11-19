# -*- coding: utf-8 -*-
"""
biogpt_gene_ranker.py
Extended version with richer evidence-based explanations.
"""

import re, math, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import sys
sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_MODEL = "microsoft/biogpt"
DEVICE = "cpu"

def extract_supporting_sentences(top_docs: List[Dict], gene: str, max_sentences: int = 5) -> List[str]:
    gene_regex = re.compile(r"\b" + re.escape(gene) + r"\b", flags=re.IGNORECASE)
    found = []
    for d in top_docs:
        for s in re.split(r'(?<=[.!?])\s+', d.get("content", "")):
            if gene_regex.search(s):
                found.append(s.strip())
                if len(found) >= max_sentences:
                    return found
    return found

class BioGPTGeneRanker:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = DEVICE, mc_samples: int = 20):
        self.model_name = model_name
        self.device = device
        self.mc_samples = mc_samples
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        print(f"Loaded model {self.model_name} on {self.device}")

    # ---------- Prompt builder ----------
    def build_prompt(self, phenotype: str, top_docs: List[Dict]) -> str:
        parts = [f"Phenotype: {phenotype}", "Relevant literature (top documents):"]
        for i, d in enumerate(top_docs, start=1):
            title = d.get("metadata", {}).get("Title") or f"Doc{i}"
            content = d.get("content", "")[:800]
            parts.append(f"[Doc{i}] {title}. {content}")
        parts.append("\nQuestion: Which of the following genes is most likely causal given the phenotype and these documents?\nAnswer:"
                    "Then, using the supporting evidence, explain mechanistically how this gene could influence the phenotype. "
                    "Provide a clear scientific summary in 5-7 sentences.\nAnswer:")
        return "\n".join(parts)

    # ---------- Candidate scoring ----------
    def score_candidate_given_prompt(self, prompt: str, candidate: str) -> float: #How well does candidate complete the prompt?
        combined = prompt + " " + candidate
        enc_combined = self.tokenizer(combined, return_tensors="pt")
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc_combined["input_ids"].to(self.device)
        attention_mask = enc_combined["attention_mask"].to(self.device)
        prompt_len = enc_prompt["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100 #
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.item()
            cand_tok = input_ids.shape[1] - prompt_len
            nll_sum = loss * max(1, cand_tok)
            return -nll_sum

    # ---------- Monte Carlo dropout ----------
    def mc_dropout_scores(self, prompt: str, candidates: List[str], T: int = None) -> Tuple[np.ndarray, np.ndarray]:
        T = T or self.mc_samples
        n = len(candidates)
        all_scores = np.zeros((T, n))
        self.model.train()
        for t in range(T):
            for i, c in enumerate(candidates):
                all_scores[t, i] = self.score_candidate_given_prompt(prompt, c)
            if (t + 1) % max(1, T // 5) == 0:
                print(f"MC dropout pass {t+1}/{T} completed.")
        self.model.eval()
        return all_scores.mean(0), all_scores.std(0)

    # ---------- NEW: richer explanation ----------
    def generate_long_explanation(self, phenotype: str, gene: str, supporting: List[str]) -> str:
        """
        Ask BioGPT to synthesize a multi-sentence explanation using retrieved evidence.
        """
        evidence_text = " ".join(supporting[:5])
        prompt = (
            f"ACCORDING TO THE RETRIEVED DOCUMENTS, {gene} is a candidate gene associated with {phenotype}. "
            f"Evidence excerpts: {evidence_text}\n\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # keep only the generated continuation
        generated = generated.replace(prompt, "").strip()
        if not generated.lower().startswith("according"):
            generated = "ACCORDING TO THE RETRIEVED DOCUMENTS, " + generated
        return generated

    # ---------- Main prediction ----------
    def predict(self, topk_docs: List[Dict], phenotype: str, candidate_genes: List[str], T: int = None) -> Dict:
        if not candidate_genes:
            raise ValueError("candidate_genes must be provided and non-empty")
        prompt = self.build_prompt(phenotype, topk_docs)
        T = T or self.mc_samples
        print("🔎 Running Monte Carlo Dropout scoring...")
        mean_scores, std_scores = self.mc_dropout_scores(prompt, candidate_genes, T=T)
        exp = np.exp(mean_scores - np.max(mean_scores))
        probs = exp / exp.sum() #Softmax to probabilities/normalize to change the number to be between 0 and 1 and sum to 1
        top_idx = int(np.argmax(probs))
        predicted_gene = candidate_genes[top_idx]
        confidence, uncertainty = float(probs[top_idx]), float(std_scores[top_idx])

        # collect evidence and synthesize explanation
        supporting = extract_supporting_sentences(topk_docs, predicted_gene, max_sentences=5)
        explanation = self.generate_long_explanation(phenotype, predicted_gene, supporting)

        explanation += (
            f"\n\nMethod note: this prediction used {T} Monte Carlo dropout forward passes "
            f"to estimate uncertainty. Confidence = {confidence:.3f}; uncertainty = {uncertainty:.3f}."
        )

        return {
            "predicted_gene": predicted_gene,
            "confidence_score": confidence,
            "mean_scores": dict(zip(candidate_genes, map(float, mean_scores))),
            "std_scores": dict(zip(candidate_genes, map(float, std_scores))),
            "probabilities": dict(zip(candidate_genes, map(float, probs))),
            "explanation": explanation,
            "top_docs_used": topk_docs[:5],
        }
