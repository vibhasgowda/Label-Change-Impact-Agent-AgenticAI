from tools.label_reader import read_document
from tools.change_detector import detect_changes
from tools.impact_assessment import assess_impact
from tools.task_generator import generate_task_list

from rag.inventory_loader import load_inventory
from rag.vector_store import VectorStore


def read_labels_node(state):
    """
    Node 1:
    Read the old and updated pharmaceutical labels.
    """

    old_text = read_document(
        state["old_label_path"]
    )

    new_text = read_document(
        state["new_label_path"]
    )

    steps = state.get(
        "agent_steps",
        []
    ).copy()

    steps.append(
        "Labels read successfully"
    )

    return {
        "old_label_text": old_text,
        "new_label_text": new_text,
        "agent_steps": steps
    }


def detect_changes_node(state):
    """
    Node 2:
    Detect meaningful safety and regulatory label changes.
    """

    result = detect_changes(
        state["old_label_text"],
        state["new_label_text"]
    )

    steps = state.get(
        "agent_steps",
        []
    ).copy()

    steps.append(
        "Safety and regulatory changes detected"
    )

    return {
        "detected_changes":
            result["change_analysis"],

        "similarity_percentage":
            result["similarity_percentage"],

        "agent_steps": steps
    }


def retrieve_documents_node(state):
    """
    Node 3:
    Search the downstream content inventory using FAISS.
    """

    documents = load_inventory(
        "data"
    )

    vector_store = VectorStore()

    vector_store.build_index(
        documents
    )

    results = vector_store.search(
        state["detected_changes"],
        top_k=4
    )

    steps = state.get(
        "agent_steps",
        []
    ).copy()

    steps.append(
        "Downstream content inventory searched"
    )

    return {
        "retrieved_documents": results,
        "agent_steps": steps
    }


def assess_impact_node(state):
    """
    Node 4:
    Determine which retrieved downstream resources
    are genuinely impacted.
    """

    result = assess_impact(
        state["detected_changes"],
        state["retrieved_documents"]
    )

    steps = state.get(
        "agent_steps",
        []
    ).copy()

    steps.append(
        "Downstream impact assessed"
    )

    return {
        "impact_assessment": result,
        "agent_steps": steps
    }


def generate_tasks_node(state):
    """
    Node 5:
    Generate the final Change-Impact Task List.
    """

    result = generate_task_list(
        state["detected_changes"],
        state["impact_assessment"]
    )

    steps = state.get(
        "agent_steps",
        []
    ).copy()

    steps.append(
        "Change-Impact Task List generated"
    )

    return {
        "task_list": result,
        "agent_steps": steps
    }