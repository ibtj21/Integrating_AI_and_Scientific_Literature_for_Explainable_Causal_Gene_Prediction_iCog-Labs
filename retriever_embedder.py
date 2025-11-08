# -*- coding: utf-8 -*-
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------
# Imports
# -----------------------------
import requests
from xml.etree import ElementTree

# ---------------------------
# Configuration
# ---------------------------

API_KEY = "6a41a7035aeb228d716b5db84d688726d908"  # <-- Replace with your PubMed (NCBI) API key
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
SEARCH_TERM = "Cancer AND GWAS AND causal gene"
MAX_RESULTS = 20

# ---------------------------
# Step 1: Search PubMed for relevant articles
# ---------------------------

search_url = f"{BASE_URL}esearch.fcgi"
search_params = {
    "db": "pubmed",
    "term": SEARCH_TERM,
    "retmax": MAX_RESULTS,
    "api_key": API_KEY,
}

response = requests.get(search_url, params=search_params)
response.raise_for_status()

# Parse XML to extract PubMed IDs
root = ElementTree.fromstring(response.content)
id_list = [id_elem.text for id_elem in root.findall(".//Id")]

print(f"✅ Retrieved {len(id_list)} PubMed IDs")

# ---------------------------
# Step 2: Fetch abstracts for the retrieved IDs
# ---------------------------

fetch_url = f"{BASE_URL}efetch.fcgi"
fetch_params = {
    "db": "pubmed",
    "id": ",".join(id_list),
    "retmode": "xml",
    "api_key": API_KEY,
}

fetch_response = requests.get(fetch_url, params=fetch_params)
fetch_response.raise_for_status()

# ---------------------------
# Step 3: Parse XML and extract article information
# ---------------------------

fetch_root = ElementTree.fromstring(fetch_response.content)
articles = []

for article in fetch_root.findall(".//PubmedArticle"):
    pmid = article.findtext(".//PMID")
    title = article.findtext(".//ArticleTitle")
    abstract = " ".join([t.text for t in article.findall(".//AbstractText") if t.text])

    if title and abstract:
        articles.append({
            "PMID": pmid,
            "Title": title.strip(),
            "Abstract": abstract.strip()
        })

# ---------------------------
# Step 4: Display results
# ---------------------------

print(f"\n🔍 Top {len(articles)} articles for '{SEARCH_TERM}':\n")
for i, art in enumerate(articles, 1):
    print(f"{i}. {art['Title']}")
    print(f"PMID: {art['PMID']}")
    print(f"Abstract: {art['Abstract'][:300]}...\n")  # Print only first 300 chars