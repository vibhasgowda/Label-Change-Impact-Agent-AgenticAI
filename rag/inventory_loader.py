from pathlib import Path


# Map folder names to downstream impact categories
CATEGORY_MAP = {
    "artwork": "Artwork",
    "website": "Website Content",
    "training": "Training Material",
    "country_submissions": "Country Submission"
}


def load_inventory(data_directory="data"):
    """
    Load downstream content inventory documents from the data folder.

    Returns documents in the format required by the FAISS vector store:
    {
        "name": "...",
        "category": "...",
        "content": "...",
        "file_path": "..."
    }
    """

    data_path = Path(data_directory)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_directory}"
        )

    documents = []

    for folder_name, category in CATEGORY_MAP.items():

        folder_path = data_path / folder_name

        if not folder_path.exists():
            continue

        # Load all TXT documents inside the category folder
        for file_path in folder_path.glob("*.txt"):

            content = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            # Ignore empty files
            if not content:
                continue

            documents.append(
                {
                    "name": file_path.stem,
                    "category": category,
                    "content": content,
                    "file_path": str(file_path)
                }
            )

    return documents