from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.1",
    temperature=0
)


def assess_impact(detected_change: str, retrieved_documents: list) -> str:
    """
    Assess which retrieved downstream resources are genuinely
    affected by a pharmaceutical label change.

    Inputs:
    - detected_change: meaningful label change identified by Tool 2
    - retrieved_documents: relevant documents returned by FAISS

    Output:
    - structured downstream impact assessment
    """

    if not detected_change:
        raise ValueError(
            "A detected label change is required."
        )

    if not retrieved_documents:
        return "NO DOWNSTREAM IMPACTS IDENTIFIED"

    # Prepare retrieved documents for Llama
    document_context = []

    for document in retrieved_documents:

        document_context.append(
            f"""
DOCUMENT NAME: {document['name']}
CATEGORY: {document['category']}
SIMILARITY SCORE: {document.get('similarity_score', 'N/A')}

DOCUMENT CONTENT:
{document['content'][:4000]}
"""
        )

    combined_documents = "\n\n---\n\n".join(
        document_context
    )

    prompt = f"""
You are the Impact Assessment Tool inside an Agentic AI system
for pharmaceutical regulatory operations.

A meaningful pharmaceutical label change has already been detected.
A semantic retrieval system has also identified potentially related
downstream resources.

Your task is to determine which retrieved resources are ACTUALLY
impacted by the detected label change.

DETECTED LABEL CHANGE:

{detected_change}

RETRIEVED DOWNSTREAM RESOURCES:

{combined_documents}

Evaluate every retrieved resource carefully.

A high semantic similarity score alone does NOT automatically mean
that a document is impacted.

A resource is impacted only when its existing content, regulatory
purpose, or required process is directly affected by the label change.

For every genuinely impacted resource, return EXACTLY:

DOCUMENT:
[Document name]

CATEGORY:
[Artwork / Website Content / Training Material / Country Submission]

IMPACT:
[Why this specific resource is affected]

PRIORITY:
[High / Medium / Low]

RECOMMENDED ACTION:
[Specific action required]

---

IMPORTANT RULES:

1. Do not invent documents.
2. Use only the retrieved resources provided.
3. Do not mark every document as impacted automatically.
4. Base the decision on the detected label change and document content.
5. Keep recommended actions specific and practical.
6. Do not offer additional assistance.
7. If none of the resources are genuinely impacted, return exactly:

NO DOWNSTREAM IMPACTS IDENTIFIED
"""

    response = llm.invoke(prompt)

    result = response.content.strip()

# Remove contradictory fallback text when actual impacts were identified
    if "DOCUMENT:" in result:
        result = result.replace(
        "NO DOWNSTREAM IMPACTS IDENTIFIED",
        ""
    ).strip()

    return result