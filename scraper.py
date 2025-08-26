import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import os
from transformers import pipeline

# -----------------------------
# 🔹 Load AI Model for Topic Classification
# -----------------------------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# You can update this topic list as needed
TOPIC_LABELS = [
    "Machine Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Robotics",
    "Bioinformatics",
    "Cybersecurity",
    "Data Science",
    "AI Ethics"
]

# -----------------------------
# 🔹 Estimate reading time based on word count
# -----------------------------
def estimate_reading_time(text):
    words = len(text.split())
    return max(1, words // 200)

# -----------------------------
# 🔹 Classpify toic using summary text
# -----------------------------
def classify_topic(summary):
    result = classifier(summary, TOPIC_LABELS)
    return result["labels"][0] if result["labels"] else "Uncategorized"

# -----------------------------
# 🔹 Scrape papers from arXiv and classify them
# -----------------------------
def scrape_arxiv(keyword, max_results=5, mode="overwrite"):
    os.makedirs("output", exist_ok=True)
    url = f"http://export.arxiv.org/api/query?search_query=all:{keyword}&start=0&max_results={max_results}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    entries = soup.find_all("entry")

    # Load existing data if appending
    existing_data = []
    if mode == "append" and os.path.exists("output/arxiv_articles.json"):
        with open("output/arxiv_articles.json", "r", encoding="utf-8") as f:
            existing_data = json.load(f)

    results = []
    for entry in entries:
        title = entry.title.text.strip()
        summary = entry.summary.text.strip()
        link = entry.id.text.strip()
        published_raw = entry.published.text.strip()
        published_date = datetime.strptime(published_raw, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
        authors = entry.find_all("author")
        first_author = authors[0].find("name").text if authors else "Unknown"
        topic = classify_topic(summary)

        content_html = f"<p>{summary}</p>"

        article = {
            "title": title,
            "date": published_date,
            "contentGroup": "Articles",
            "internalTags": [topic],
            "author": {
                "name": first_author,
                "email": "",
                "organization": ""
            },
            "publication": {
                "name": "arXiv",
                "url": link
            },
            "publicTags": [],
            "summary": summary,
            "sourceUrl": link,
            "language": "en",
            "readingTime": estimate_reading_time(summary),
            "imageUrl": "",
            "relatedContent": [],
            "content": content_html
        }

        # Avoid duplicates
        if not any(existing["title"] == article["title"] for existing in existing_data):
            results.append(article)

    all_data = existing_data + results

    # Save the combined result
    with open("output/arxiv_articles.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} new articles (total: {len(all_data)}) to output/arxiv_articles.json")

# -----------------------------
# 🔹 Entry point for user interaction
# -----------------------------
if __name__ == "__main__":
    print("Research Paper Scraper with AI Topic Classification\n")

    kw = input("Enter a keyword to search for research papers: ")
    max_papers = input("How many papers do you want to fetch? (e.g., 5, 10): ")

    try:
        max_results = int(max_papers)
    except ValueError:
        print("Invalid number. Using default (5).")
        max_results = 5

    mode = input("Do you want to overwrite the existing JSON or append to it? (overwrite/append): ").lower()
    if mode not in ["overwrite", "append"]:
        print("Invalid option. Defaulting to 'overwrite'")
        mode = "overwrite"

    scrape_arxiv(kw, max_results=max_results, mode=mode)
