
"""
Post-Processing & Vector Indexing Script
Final step to ensure all scraped and synthetic data is indexed for the RAG API.
"""
import os
import json
from utils.vector_db import vector_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indexing")

def index_production_data():
    data_file = "data/processed.jsonl"
    if not os.path.exists(data_file):
        logger.error("No processed data found. Run data generation/scraping first.")
        return

    logger.info("Indexing documentation for RAG...")
    documents = []
    ids = []
    
    with open(data_file, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            # Only index documentation-style entries for RAG
            if "React" in item["text"]:
                documents.append(item["text"])
                ids.append(f"doc_{i}")
    
    if documents:
        vector_db.add_documents(documents, ids)
        logger.info(f"✅ Indexed {len(documents)} document snippets for production RAG.")
    else:
        logger.warning("No React documentation found in data file to index.")

if __name__ == "__main__":
    index_production_data()
