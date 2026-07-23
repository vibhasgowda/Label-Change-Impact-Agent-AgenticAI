from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (
    read_labels_node,
    detect_changes_node,
    retrieve_documents_node,
    assess_impact_node,
    generate_tasks_node,
)


def create_label_change_agent():
    """
    Build and compile the Label Change Impact Agent workflow.
    """

    workflow = StateGraph(AgentState)

    # Add agent workflow nodes
    workflow.add_node(
        "read_labels",
        read_labels_node
    )

    workflow.add_node(
        "detect_changes",
        detect_changes_node
    )

    workflow.add_node(
        "retrieve_documents",
        retrieve_documents_node
    )

    workflow.add_node(
        "assess_impact",
        assess_impact_node
    )

    workflow.add_node(
        "generate_tasks",
        generate_tasks_node
    )

    # Define workflow sequence
    workflow.add_edge(
        START,
        "read_labels"
    )

    workflow.add_edge(
        "read_labels",
        "detect_changes"
    )

    workflow.add_edge(
        "detect_changes",
        "retrieve_documents"
    )

    workflow.add_edge(
        "retrieve_documents",
        "assess_impact"
    )

    workflow.add_edge(
        "assess_impact",
        "generate_tasks"
    )

    workflow.add_edge(
        "generate_tasks",
        END
    )

    # Compile the LangGraph agent
    return workflow.compile()


# Ready-to-use compiled agent
label_change_agent = create_label_change_agent()