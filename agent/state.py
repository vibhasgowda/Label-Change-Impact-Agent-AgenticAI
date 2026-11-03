
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """
    Shared state for the Agentic Pharmaceutical
    Label Change Impact Assessment system.
    """

    # --------------------------------------------------
    # 1. INPUT DOCUMENTS
    # --------------------------------------------------

    old_label_path: str
    new_label_path: str

    # --------------------------------------------------
    # 2. EXTRACTED LABEL CONTENT
    # --------------------------------------------------

    old_label_text: str
    new_label_text: str

    # --------------------------------------------------
    # 3. CHANGE DETECTION
    # --------------------------------------------------

    detected_changes: str
    changed_sections: List[Dict[str, Any]]
    similarity_percentage: float
    meaningful_change_detected: bool

    # --------------------------------------------------
    # 4. RETRIEVAL AND RAG
    # --------------------------------------------------

    retrieved_documents: List[Dict[str, Any]]
    retrieval_query: str
    retrieval_attempts: int
    max_retrieval_attempts: int
    retrieval_sufficient: bool
    retrieval_feedback: str

    # --------------------------------------------------
    # 5. IMPACT ASSESSMENT
    # --------------------------------------------------

    impact_assessment: str
    assessment_attempts: int
    assessment_valid: bool
    assessment_feedback: str

    # --------------------------------------------------
    # 6. TASK GENERATION AND VALIDATION
    # --------------------------------------------------

    task_list: str
    task_validation_passed: bool
    task_validation_feedback: str

    # --------------------------------------------------
    # 7. HUMAN REVIEW
    # --------------------------------------------------

    human_review_required: bool
    human_approved: Optional[bool]
    review_status: str

    # --------------------------------------------------
    # 8. AGENT DECISION TRACKING
    # --------------------------------------------------

    agent_steps: List[str]
    agent_decisions: List[str]
    error_message: Optional[str]
