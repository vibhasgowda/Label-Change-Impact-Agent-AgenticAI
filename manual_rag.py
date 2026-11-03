from rag.inventory_loader import load_inventory
from rag.vector_store import VectorStore


print("RAG RETRIEVAL TEST STARTED")

try:
    # Load downstream content inventory
    documents = load_inventory("data")

    print(f"\nLoaded {len(documents)} downstream documents.")

    for document in documents:
        print(
            f"- {document['name']} "
            f"({document['category']})"
        )

    # Build FAISS vector database
    print("\nBuilding FAISS vector store...")

    vector_store = VectorStore()
    vector_store.build_index(documents)

    print("FAISS vector store built successfully.")

    # Test with an actual safety change detected in our label
    query = """
    A new requirement has been added to monitor renal function
    every 6 months in patients at increased risk of renal impairment
    while using Metformin Hydrochloride Tablets.
    """

    print("\nSearching for affected downstream resources...")
    print(f"\nDetected Change:\n{query.strip()}")

    results = vector_store.search(
        query,
        top_k=4
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for number, result in enumerate(results, start=1):

        print(f"\nResult {number}")
        print(f"Document: {result['name']}")
        print(f"Category: {result['category']}")
        print(
            f"Similarity Score: "
            f"{result['similarity_score']}"
        )

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL TEST COMPLETED")
    print("=" * 60)

except Exception as error:
    print(f"\nError: {error}")