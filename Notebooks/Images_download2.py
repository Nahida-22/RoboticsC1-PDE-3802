import os
import time
import requests
from tqdm import tqdm
from ddgs import DDGS

BASE_DIR = os.getcwd()
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "raw")
CATEGORIES = ["paperClip", "glueStick", "stapler", "pencilBox"]
IMAGES_PER_CATEGORY = 1000

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def download_images(query, folder, limit=1000):
    """Download images from DuckDuckGo search with resume support."""
    os.makedirs(folder, exist_ok=True)
    
    # Count already downloaded images
    existing_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    start_index = len(existing_files)
    remaining = limit - start_index

    if remaining <= 0:
        print(f" Already have {limit} images for {query}. Skipping.\n")
        return

    print(f"\n Searching for '{query}' images... ({start_index}/{limit} already downloaded)")

    ddgs = DDGS()
    results = ddgs.images(query, max_results=limit * 2)  # fetch more to ensure we can skip duplicates

    count = start_index
    seen_urls = set()

    # Avoid duplicates: read existing file URLs if we saved them before
    urls_log_path = os.path.join(folder, "_urls.txt")
    if os.path.exists(urls_log_path):
        with open(urls_log_path, "r", encoding="utf-8") as f:
            seen_urls.update(f.read().splitlines())

    with open(urls_log_path, "a", encoding="utf-8") as log:
        for r in tqdm(results, desc=f"Downloading {query}", total=limit):
            url = r.get("image")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200 and len(response.content) > 10240:  # >10KB
                    ext = os.path.splitext(url)[1].split("?")[0] or ".jpg"
                    if len(ext) > 5:
                        ext = ".jpg"
                    filepath = os.path.join(folder, f"{query}_{count+1}{ext}")
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    log.write(url + "\n")
                    count += 1
            except Exception:
                continue

            if count >= limit:
                break

    print(f" Finished downloading {count} images for {query}.\n")

if __name__ == "__main__":
    for category in CATEGORIES:
        category_folder = os.path.join(OUTPUT_BASE_DIR, category)
        download_images(category, category_folder, IMAGES_PER_CATEGORY)
        time.sleep(2)
