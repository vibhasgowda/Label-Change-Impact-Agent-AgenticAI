from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    """
    Shared state passed between all nodes of the
    Label Change Impact Agent.
    """

    # Input file paths
    old_label_path: str
    new_label_path: str

    # Extracted label text
    old_label_text: str
    new_label_text: str

    # Change detection results
    detected_changes: str
    similarity_percentage: float

    # RAG retrieval results
    retrieved_documents: List[Dict[str, Any]]

    # Impact assessment result
    impact_assessment: str

    # Final NVIDIA deliverable
    task_list: str

    # Workflow tracking for the UI
    agent_steps: List[str]