import gzip
import json
import re
import hashlib
import requests
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FILE = "astronomy_clean.txt"

MIN_WORDS_REQUIRED = 100_000

# Minimum acceptable document length
MIN_DOCUMENT_WORDS = 40

# Maximum document length we allow
MAX_DOCUMENT_WORDS = 20_000

DOWNLOAD_FOLDER = Path("haina_astronomy_files")


FILES = {
    "HainaWeb-Sci_Astronomy_00318.jsonl.gz":
        "https://huggingface.co/datasets/anonymous-eandd-2026/HainaWeb-Sci-sample/resolve/main/Astronomy/HainaWeb-Sci_Astronomy_00318.jsonl.gz",

    "HainaWeb-Sci_Astronomy_00733.jsonl.gz":
        "https://huggingface.co/datasets/anonymous-eandd-2026/HainaWeb-Sci-sample/resolve/main/Astronomy/HainaWeb-Sci_Astronomy_00733.jsonl.gz"
}


# ============================================================
# DOWNLOAD FILES
# ============================================================

def download_file(url, destination):

    if destination.exists():
        print(f"[Already exists] {destination}")
        return

    print(f"\nDownloading: {destination.name}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_bytes = 0

    with open(destination, "wb") as file:

        for chunk in response.iter_content(chunk_size=1024 * 1024):

            if chunk:
                file.write(chunk)
                total_bytes += len(chunk)

                print(
                    f"\rDownloaded: {total_bytes / (1024 * 1024):.2f} MB",
                    end=""
                )

    print("\nDownload completed.")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return None

    # Normalize whitespace
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # Remove HTML-like tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove excessive repeated punctuation
    text = re.sub(r"([!?.,])\1{2,}", r"\1", text)

    # Normalize spaces again
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# QUALITY FILTERING
# ============================================================

def is_good_document(text):

    if not text:
        return False

    words = text.split()

    word_count = len(words)

    # Too short
    if word_count < MIN_DOCUMENT_WORDS:
        return False

    # Extremely long pages may contain navigation dumps
    if word_count > MAX_DOCUMENT_WORDS:
        return False

    # Must contain enough alphabetic characters
    alphabetic_characters = sum(
        character.isalpha()
        for character in text
    )

    if alphabetic_characters < 100:
        return False

    # Reject text with excessive non-letter symbols
    symbol_count = sum(
        not character.isalnum() and not character.isspace()
        for character in text
    )

    if symbol_count / max(len(text), 1) > 0.25:
        return False

    # Reject pages that look like navigation/menu garbage
    garbage_patterns = [
        "cookie policy",
        "accept all cookies",
        "sign in register",
        "javascript required",
        "enable javascript",
        "privacy policy terms",
        "all rights reserved",
        "skip to content",
        "subscribe to newsletter",
        "log in sign up"
    ]

    text_lower = text.lower()

    garbage_matches = sum(
        phrase in text_lower
        for phrase in garbage_patterns
    )

    if garbage_matches >= 3:
        return False

    return True


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def get_text_hash(text):

    """
    Creates a unique fingerprint for a document.
    Duplicate documents will have the same hash.
    """

    normalized = text.lower().strip()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# PROCESS ONE JSONL FILE
# ============================================================

def process_file(file_path, seen_hashes, output_file):

    total_records = 0
    accepted_records = 0
    rejected_records = 0
    duplicate_records = 0
    total_words = 0

    print(f"\nProcessing: {file_path.name}")

    with gzip.open(file_path, "rt", encoding="utf-8") as file:

        for line in file:

            total_records += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                rejected_records += 1
                continue

            # Extract main text field
            text = record.get("text", "")

            # Clean text
            text = clean_text(text)

            # Quality filtering
            if not is_good_document(text):
                rejected_records += 1
                continue

            # Duplicate detection
            text_hash = get_text_hash(text)

            if text_hash in seen_hashes:
                duplicate_records += 1
                continue

            seen_hashes.add(text_hash)

            # Save document
            output_file.write(text + "\n\n")

            accepted_records += 1
            total_words += len(text.split())

            if accepted_records % 100 == 0:

                print(
                    f"\rAccepted: {accepted_records:,} | "
                    f"Words: {total_words:,}",
                    end=""
                )

    print("\n")

    print(f"Total records:       {total_records:,}")
    print(f"Accepted records:    {accepted_records:,}")
    print(f"Rejected records:    {rejected_records:,}")
    print(f"Duplicates removed:  {duplicate_records:,}")
    print(f"Words extracted:     {total_words:,}")

    return total_words


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("HainaWeb Astronomy Dataset Cleaner")
    print("=" * 60)

    DOWNLOAD_FOLDER.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Download files
    # --------------------------------------------------------

    for filename, url in FILES.items():

        destination = DOWNLOAD_FOLDER / filename

        download_file(url, destination)

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    seen_hashes = set()

    total_words = 0

    print("\nCreating clean text file...\n")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for filename in FILES:

            file_path = DOWNLOAD_FOLDER / filename

            words_from_file = process_file(
                file_path,
                seen_hashes,
                output_file
            )

            total_words += words_from_file

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL DATASET REPORT")
    print("=" * 60)

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total unique documents: {len(seen_hashes):,}")
    print(f"Total words: {total_words:,}")

    if total_words >= MIN_WORDS_REQUIRED:

        print("\nSUCCESS!")
        print(
            f"The dataset contains at least "
            f"{MIN_WORDS_REQUIRED:,} words."
        )

    else:

        print("\nWARNING!")
        print(
            f"The dataset contains only {total_words:,} words."
        )

        print(
            "You may need to lower filtering restrictions "
            "or obtain additional astronomy data."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()