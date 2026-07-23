from rag.inventory_loader import load_inventory
from rag.vector_store import VectorStore
from tools.impact_assessment import assess_impact


print("IMPACT ASSESSMENT TEST STARTED")

try:
    # Test with a meaningful change detected by Tool 2
    detected_change = """
    A new requirement has been added to monitor renal function
    every 6 months in patients at increased risk of renal impairment
    while using Metformin Hydrochloride Tablets.
    """

    # Load downstream inventory
    documents = load_inventory("data")

    print(f"\nLoaded {len(documents)} downstream documents.")

    # Build FAISS vector store
    print("Building FAISS vector store...")

    vector_store = VectorStore()
    vector_store.build_index(documents)

    print("FAISS vector store built successfully.")

    # Retrieve potentially affected documents
    print("\nRetrieving potentially affected resources...")

    retrieved_documents = vector_store.search(
        detected_change,
        top_k=4
    )

    print(
        f"Retrieved {len(retrieved_documents)} "
        "potentially related resources."
    )

    # Let Tool 4 determine actual impact
    print(
        "\nAssessing actual downstream impact "
        "with Llama 3.1... Please wait.\n"
    )

    result = assess_impact(
        detected_change,
        retrieved_documents
    )

    print("=" * 60)
    print("DOWNSTREAM IMPACT ASSESSMENT")
    print("=" * 60)

    print(result)

    print("\n" + "=" * 60)
    print("IMPACT ASSESSMENT COMPLETED")
    print("=" * 60)

except Exception as error:
    print(f"\nError: {error}")