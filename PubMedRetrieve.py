# -*- coding: utf-8 -*-
import sys
import time
import requests
from xml.etree import ElementTree
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')


class PubMedRetriever:
    """
    A lightweight, modular class to search and retrieve PubMed abstracts using NCBI E-utilities.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or "6a41a7035aeb228d716b5db84d688726d908"
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.headers = {
            "User-Agent": "HannaGenePredictor/1.0 (contact: your_email@example.com)"
        }

    # ---------------------------
    # Step 1: Search PubMed
    # ---------------------------
    def search_pubmed(self, term, max_results=20):
        """Search PubMed and return a list of PMIDs."""
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": max_results,
            "api_key": self.api_key,
        }

        print(f"🔎 Searching PubMed for: '{term}' ...")
        response = requests.get(search_url, params=params, headers=self.headers, timeout=15)
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)
        id_list = [id_elem.text for id_elem in root.findall(".//Id")]

        print(f"✅ Retrieved {len(id_list)} PubMed IDs.")
        return id_list

    # ---------------------------
    # Step 2: Fetch Abstracts
    # ---------------------------
    def fetch_abstracts(self, id_list, batch_size=5):
        """Fetch abstracts from PubMed in small batches with retries."""
        all_articles = []
        fetch_url = f"{self.base_url}efetch.fcgi"

        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "retmode": "xml",
                "api_key": self.api_key,
            }

            success = False
            for attempt in range(3):
                try:
                    print(f"📥 Fetching batch {i//batch_size + 1} (IDs {i+1}-{i+len(batch_ids)})...")
                    response = requests.get(fetch_url, params=params, headers=self.headers, timeout=120)
                    response.raise_for_status()

                    root = ElementTree.fromstring(response.content)
                    for article in root.findall(".//PubmedArticle"):
                        pmid = article.findtext(".//PMID")
                        title = article.findtext(".//ArticleTitle")
                        abstract = " ".join([t.text for t in article.findall(".//AbstractText") if t.text])
                        if title and abstract:
                            all_articles.append({
                                "PMID": pmid,
                                "Title": title.strip(),
                                "Abstract": abstract.strip()
                            })
                    success = True
                    break

                except requests.exceptions.Timeout:
                    print(f"⚠️ Timeout on batch {i//batch_size + 1} (attempt {attempt+1}/3), retrying...")
                    time.sleep(5)
                except Exception as e:
                    print(f"❌ Error fetching batch {i//batch_size + 1}: {e}")
                    time.sleep(5)

            if not success:
                print(f"🚫 Failed to fetch batch {i//batch_size + 1} after 3 attempts.")

        print(f"✅ Retrieved {len(all_articles)} abstracts successfully.")
        return all_articles

    # ---------------------------
    # Step 3: Full Retrieval Pipeline
    # ---------------------------
    def retrieve(self, search_term, max_results=20):
        """Perform full pipeline: search → fetch → return DataFrame."""
        ids = self.search_pubmed(search_term, max_results)
        time.sleep(1)
        articles = self.fetch_abstracts(ids)
        if not articles:
            print("⚠️ No abstracts retrieved.")
            return pd.DataFrame()
        df = pd.DataFrame(articles)
        print(f"✅ Retrieved {len(df)} abstracts and loaded into memory.")
        return df
