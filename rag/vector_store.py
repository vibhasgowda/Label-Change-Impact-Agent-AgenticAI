import faiss
import numpy as np

from rag.embeddings import EmbeddingModel


class VectorStore:
    """
    FAISS-based semantic search for downstream pharmaceutical
    resources such as artwork, website content, training materials,
    and country submission documents.
    """

    def __init__(self):

        self.embedding_model = EmbeddingModel()

        self.index = None

        self.documents = []


    def build_index(self, documents):
        """
        Build a FAISS index from downstream document records.

        Each document should contain:
        {
            "name": "...",
            "category": "...",
            "content": "..."
        }
        """

        if not documents:
            raise ValueError(
                "No documents were provided to build the vector store."
            )

        self.documents = documents

        texts = [
            document["content"]
            for document in documents
        ]

        embeddings = self.embedding_model.embed_documents(
            texts
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        # Inner product works as cosine similarity because
        # embeddings are already normalized.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )


    def search(self, query, top_k=5):
        """
        Search the knowledge base for documents semantically
        related to a detected pharmaceutical label change.
        """

        if self.index is None:

            raise ValueError(
                "Vector store has not been built yet."
            )

        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            min(
                top_k,
                len(self.documents)
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            document = self.documents[
                index
            ].copy()

            document["similarity_score"] = round(
                float(score),
                4
            )

            results.append(
                document
            )

        return results