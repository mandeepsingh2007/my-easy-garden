import os
import csv
import time
import hashlib
import random
import string
import urllib.parse
import requests

CSV_PATH = "empty_places_prompts.csv"   # path to your prompts CSV (class,prompt)
OUT_DIR  = "out"                         # where to save images
DELAY_S  = 0.5                           # polite delay between requests
RETRIES  = 3                             # retry attempts per image

# Optional: per-class image count cap (None = download all in CSV for that class)
MAX_PER_CLASS = None  # e.g., set to 1000 to enforce 1000/class

def safe_name(s, max_len=64):
    """Make a safe filesystem-friendly chunk from a string."""
    keep = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned = "".join(c if c in keep else "_" for c in s)
    return cleaned[:max_len]

def pollinations_url(prompt: str) -> str:
    """
    Build the Pollinations endpoint.
    The user-provided pattern is: https://pollinations.ai/p/<description>
    We’ll URL-encode the prompt safely.
    """
    return "https://pollinations.ai/p/" + urllib.parse.quote_plus(prompt)

def is_image_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "")
    return ctype.startswith("image/")

def download_image(prompt: str, out_path: str) -> bool:
    """
    Download a single image for the prompt to out_path.
    Returns True if saved successfully, False otherwise.
    """
    url = pollinations_url(prompt)
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code == 200 and is_image_response(r):
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                # Non-image or non-200; backoff a bit and retry
                time.sleep(DELAY_S * attempt)
        except requests.RequestException:
            time.sleep(DELAY_S * attempt)
    return False

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Cannot find CSV at '{CSV_PATH}'. "
            "Expected a file with headers: class,prompt"
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    # Track counts per class if you want to cap
    per_class_counts = {}

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "class" not in reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError("CSV must have headers: class,prompt")

        for row in reader:
            cls = row["class"].strip()
            prompt = row["prompt"].strip()

            if not cls or not prompt:
                continue

            # Respect per-class limit if set
            if MAX_PER_CLASS is not None:
                c = per_class_counts.get(cls, 0)
                if c >= MAX_PER_CLASS:
                    continue

            # Create class directory
            class_dir = os.path.join(OUT_DIR, safe_name(cls))
            os.makedirs(class_dir, exist_ok=True)

            # Build a stable, short filename (hash of prompt to avoid dupes)
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
            base = safe_name(prompt, max_len=40)
            filename = f"{base}_{h}.jpg"
            out_path = os.path.join(class_dir, filename)

            # Skip if already exists
            if os.path.exists(out_path):
                per_class_counts[cls] = per_class_counts.get(cls, 0) + 1
                continue

            ok = download_image(prompt, out_path)
            if ok:
                per_class_counts[cls] = per_class_counts.get(cls, 0) + 1
                print(f"[OK] {cls}: {out_path}")
            else:
                print(f"[FAIL] {cls}: could not fetch image for prompt -> {prompt}")

            time.sleep(DELAY_S)  # be polite

    print("\nDone.")
    for cls, cnt in sorted(per_class_counts.items()):
        print(f"{cls}: {cnt} images")

if __name__ == "__main__":
    main()
