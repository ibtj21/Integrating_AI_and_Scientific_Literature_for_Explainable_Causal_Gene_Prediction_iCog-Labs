# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class CausalGenePredictor:
    """
    Predicts the likely causal gene using BioGPT given a phenotype and top documents.
    """

    def __init__(self, model_name="microsoft/biogpt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading BioGPT model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    @staticmethod
    def clean_reason(text, max_words=25):
        words = text.split()
        return " ".join(words[:max_words]).strip()

    def build_prompt(self, phenotype: str, genes: list, docs: list) -> str:
        """Construct a prompt including top 5 retrieved documents."""
        genes_str = ", ".join(f"{{{g}}}" for g in genes)
        docs_text = "\n".join([f"- {d['content'][:300]}..." for d in docs])  # truncate long docs
        prompt = f"""You are an expert in genetics. Identify the likely causal gene using the evidence below.
Example:
GWAS phenotype: Type 2 Diabetes
Genes: {{TCF7L2}}, {{KCNJ11}}, {{PPARG}}
Answer: causal_gene: TCF7L2, confidence: 0.85, reason: Strongest replicated gene across multiple GWAS studies.
---
Evidence from top 5 documents:
{docs_text}

Now your turn:
GWAS phenotype: {phenotype}
Genes: {genes_str}
Answer: causal_gene:"""
        return prompt

    def predict(self, phenotype: str, genes: list, top_docs: list) -> dict:
        prompt = self.build_prompt(phenotype, genes, top_docs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        raw_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        answer_block = raw_response.split("Now your turn:")[-1] if "Now your turn:" in raw_response else raw_response

        # Regex for structured answer
        match = re.search(
            r"causal\s*[_ ]?\s*gene[: ]+([A-Za-z0-9_-]+).*?"
            r"confidence[: ]+([0-9.]+).*?"
            r"reason[: ]+(.+?)(?=(?:\b(?:Genes|Answer)\b|$))",
            answer_block,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            prediction = {
                "causal_gene": match.group(1).strip(),
                "confidence": float(match.group(2)),
                "reason": self.clean_reason(match.group(3))
            }
        else:
            fallback = re.search(r"(" + "|".join(genes) + ")", answer_block)
            prediction = {
                "causal_gene": fallback.group(1) if fallback else None,
                "confidence": None,
                "reason": self.clean_reason(answer_block)
            }

        # Further clean reason
        if prediction["reason"]:
            reason_parts = re.split(r'reason:|, confidence:|$', prediction["reason"])
            cleaned_reason = next((part.strip(" .,:;-") for part in reason_parts if part.strip()), "")
            prediction["reason"] = cleaned_reason

        return prediction
