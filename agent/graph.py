
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from agent.state import AgentState
from agent.nodes import (
    read_labels_node,
    detect_changes_node,
    retrieve_documents_node,
    assess_impact_node,
    generate_tasks_node,
)


# --------------------------------------------------
# ROUTING DECISIONS
# --------------------------------------------------

def route_after_detection(state: AgentState):
    if state.get("meaningful_change_detected", False):
        return "retrieve_documents"

    return "no_change_result"


def route_after_retrieval(state: AgentState):
    if state.get("retrieval_sufficient", False):
        return "assess_impact"

    attempts = state.get("retrieval_attempts", 0)
    max_attempts = state.get("max_retrieval_attempts", 2)

    if attempts < max_attempts:
        return "retrieve_documents"

    return "insufficient_evidence_result"


def route_after_assessment(state: AgentState):
    if state.get("assessment_valid", False):
        return "generate_tasks"

    return "assessment_review_required"


def route_after_task_generation(state: AgentState):
    if state.get("task_validation_passed", False):
        return "human_review"

    return "task_review_required"


# --------------------------------------------------
# RESULT AND REVIEW NODES
# --------------------------------------------------

def no_change_result_node(state: AgentState):
    steps = state.get("agent_steps", []).copy()
    steps.append(
        "No meaningful safety change; impact assessment skipped"
    )

    return {
        "impact_assessment": (
            "No meaningful safety or regulatory changes detected. "
            "Downstream impact assessment was not performed."
        ),
        "task_list": "NO DOWNSTREAM CHANGE-IMPACT TASKS GENERATED.",
        "human_review_required": True,
        "review_status": "Pending human review",
        "agent_steps": steps,
        "agent_decisions": state.get("agent_decisions", []).copy()
        + ["No-change route selected"]
    }


def insufficient_evidence_result_node(state: AgentState):
    steps = state.get("agent_steps", []).copy()
    steps.append(
        "Retrieval limit reached; manual evidence review required"
    )

    return {
        "impact_assessment": (
            "INSUFFICIENT EVIDENCE: The agent could not retrieve "
            "sufficient downstream documentation. No definitive "
            "impact conclusion should be made."
        ),
        "task_list": (
            "MANUAL REVIEW REQUIRED: Identify missing downstream "
            "documents and complete the impact assessment."
        ),
        "human_review_required": True,
        "review_status": "Pending manual evidence review",
        "agent_steps": steps
    }


def assessment_review_required_node(state: AgentState):
    steps = state.get("agent_steps", []).copy()
    steps.append("Impact assessment validation failed")

    return {
        "human_review_required": True,
        "review_status": "Pending manual impact assessment verification",
        "agent_steps": steps
    }


def human_review_node(state: AgentState):
    """
    Pause execution and wait for an explicit human decision.
    Resume using Command(resume={...}).
    """

    decision = interrupt({
        "type": "human_approval",
        "message": (
            "Review the generated change-impact tasks "
            "and approve or reject them."
        ),
        "task_list": state.get("task_list", ""),
        "impact_assessment": state.get("impact_assessment", "")
    })

    approved = (
        isinstance(decision, dict)
        and decision.get("approved") is True
    )

    steps = state.get("agent_steps", []).copy()
    decisions = state.get("agent_decisions", []).copy()

    if approved:
        steps.append("Human reviewed and approved the generated tasks")
        decisions.append("Human approval granted")

        status = "Approved by human reviewer"

    else:
        steps.append("Human reviewer rejected the generated tasks")
        decisions.append("Human approval rejected")

        status = "Rejected by human reviewer"

    return {
        "human_review_required": True,
        "human_approved": approved,
        "review_status": status,
        "agent_steps": steps,
        "agent_decisions": decisions
    }


def task_review_required_node(state: AgentState):
    steps = state.get("agent_steps", []).copy()
    steps.append(
        "Task validation failed; manual verification required"
    )

    return {
        "human_review_required": True,
        "review_status": "Pending manual task verification",
        "agent_steps": steps
    }


# --------------------------------------------------
# GRAPH CONSTRUCTION
# --------------------------------------------------

def create_label_change_agent():

    workflow = StateGraph(AgentState)

    # Processing nodes
    workflow.add_node("read_labels", read_labels_node)
    workflow.add_node("detect_changes", detect_changes_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("assess_impact", assess_impact_node)
    workflow.add_node("generate_tasks", generate_tasks_node)

    # Decision and review nodes
    workflow.add_node("no_change_result", no_change_result_node)

    workflow.add_node(
        "insufficient_evidence_result",
        insufficient_evidence_result_node
    )

    workflow.add_node(
        "assessment_review_required",
        assessment_review_required_node
    )

    workflow.add_node("human_review", human_review_node)

    workflow.add_node(
        "task_review_required",
        task_review_required_node
    )

    # Initial workflow
    workflow.add_edge(START, "read_labels")
    workflow.add_edge("read_labels", "detect_changes")

    # Decision 1: meaningful change?
    workflow.add_conditional_edges(
        "detect_changes",
        route_after_detection,
        {
            "retrieve_documents": "retrieve_documents",
            "no_change_result": "no_change_result"
        }
    )

    # Decision 2: sufficient retrieval evidence?
    workflow.add_conditional_edges(
        "retrieve_documents",
        route_after_retrieval,
        {
            "retrieve_documents": "retrieve_documents",
            "assess_impact": "assess_impact",
            "insufficient_evidence_result": "insufficient_evidence_result"
        }
    )

    # Decision 3: valid impact assessment?
    workflow.add_conditional_edges(
        "assess_impact",
        route_after_assessment,
        {
            "generate_tasks": "generate_tasks",
            "assessment_review_required": "assessment_review_required"
        }
    )

    # Decision 4: valid generated tasks?
    workflow.add_conditional_edges(
        "generate_tasks",
        route_after_task_generation,
        {
            "human_review": "human_review",
            "task_review_required": "task_review_required"
        }
    )

    # Terminal paths
    workflow.add_edge("no_change_result", END)
    workflow.add_edge("insufficient_evidence_result", END)
    workflow.add_edge("assessment_review_required", END)
    workflow.add_edge("human_review", END)
    workflow.add_edge("task_review_required", END)

    # Checkpointing enables pause and resume
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# Compiled agent
label_change_agent = create_label_change_agent()
