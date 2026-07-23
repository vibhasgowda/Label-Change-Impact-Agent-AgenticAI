from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Creates semantic embeddings for downstream pharmaceutical
    documents and detected label changes.
    """

    def __init__(self):
        # Lightweight and reliable sentence-transformer model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        """
        Generate embeddings for multiple document chunks.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def embed_query(self, query):
        """
        Generate an embedding for a detected label change
        used to search the downstream document inventory.
        """

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding