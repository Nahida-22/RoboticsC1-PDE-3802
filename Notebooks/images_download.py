from bing_image_downloader import downloader
import os

OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "raw")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

CATEGORIES = [ "pencilBox"]

for category in CATEGORIES:
    folder = os.path.join(OUTPUT_BASE_DIR, category)
    print(f"🔍 Downloading images for {category} ...")
    downloader.download(category, limit=1000,  output_dir=OUTPUT_BASE_DIR, adult_filter_off=True, force_replace=False, timeout=60)
    print(f"✅ Done: {category}\n")
