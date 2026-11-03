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
    Detect and classify pharmaceutical label changes.
    Distinguish textual differences from meaningful
    regulatory or safety changes.
    """

    result = detect_changes(
        state["old_label_text"],
        state["new_label_text"]
    )

    changed_sections = result.get("changed_sections", [])

    change_analysis = result.get("change_analysis", "")

    meaningful_change = (
        bool(changed_sections)
        and "NO MEANINGFUL SAFETY CHANGES DETECTED"
        not in change_analysis.upper()
    )

    steps = state.get("agent_steps", []).copy()
    steps.append("Label changes detected and classified")

    decisions = state.get("agent_decisions", []).copy()

    if meaningful_change:
        decisions.append(
            f"Meaningful regulatory or safety changes detected "
            f"in {len(changed_sections)} section(s)"
        )
    else:
        decisions.append(
            "No meaningful safety changes detected; "
            "downstream impact assessment can be skipped"
        )

    return {
        "detected_changes": change_analysis,
        "changed_sections": changed_sections,
        "similarity_percentage": result["similarity_percentage"],
        "meaningful_change_detected": meaningful_change,
        "agent_steps": steps,
        "agent_decisions": decisions
    }



def retrieve_documents_node(state):
    """
    Retrieve downstream documents and evaluate whether
    the retrieved evidence is sufficient.

    Retrieval attempts are bounded by the graph router.
    """

    import re

    from rag.inventory_loader import load_inventory
    from rag.vector_store import VectorStore

    attempts = state.get("retrieval_attempts", 0) + 1
    max_attempts = state.get("max_retrieval_attempts", 2)

    changed_sections = state.get("changed_sections", [])

    # Build a focused retrieval query.
    if attempts == 1 or not state.get("retrieval_query"):

        section_names = [
            item.get("section", "")
            for item in changed_sections
        ]

        retrieval_query = (
            state.get("detected_changes", "")
            + " "
            + " ".join(section_names)
        )

    else:

        retrieval_query = (
            state.get("retrieval_query", "")
            + " pharmaceutical downstream impact "
            + "artwork website training country submission"
        )

    # Load downstream inventory.
    documents = load_inventory("data")

    if not documents:

        retrieved_documents = []
        retrieval_sufficient = False

    else:

        vector_store = VectorStore()
        vector_store.build_index(documents)

        retrieved_documents = vector_store.search(
            retrieval_query,
            top_k=4
        )

        # Require at least one relevant result.
        # Threshold will be calibrated using test data.
        relevant_documents = [
            document
            for document in retrieved_documents
            if document.get("similarity_score", 0) >= 0.30
        ]

        retrieved_documents = relevant_documents

        retrieval_sufficient = len(retrieved_documents) > 0

    steps = state.get("agent_steps", []).copy()

    steps.append(
        f"Downstream inventory retrieval attempt "
        f"{attempts}/{max_attempts} completed"
    )

    decisions = state.get("agent_decisions", []).copy()

    if retrieval_sufficient:

        decisions.append(
            f"Retrieved {len(retrieved_documents)} relevant "
            f"downstream document(s)"
        )

    else:

        decisions.append(
            "Retrieved evidence insufficient; "
            "agent evaluates whether another search is possible"
        )

    return {
        "retrieved_documents": retrieved_documents,
        "retrieval_query": retrieval_query,
        "retrieval_attempts": attempts,
        "max_retrieval_attempts": max_attempts,
        "retrieval_sufficient": retrieval_sufficient,
        "retrieval_feedback": (
            "Relevant documents retrieved"
            if retrieval_sufficient
            else "No documents met the current relevance threshold"
        ),
        "agent_steps": steps,
        "agent_decisions": decisions
    }




def assess_impact_node(state):
    """
    Assess downstream impacts and validate the assessment
    against retrieved evidence.
    """

    import re

    retrieved_documents = state.get("retrieved_documents", [])

    result = assess_impact(
        state["detected_changes"],
        retrieved_documents
    )

    assessment = result.strip()

    attempts = state.get("assessment_attempts", 0) + 1

    steps = state.get("agent_steps", []).copy()
    steps.append("Downstream impact assessment completed")

    decisions = state.get("agent_decisions", []).copy()

    if assessment.upper() == "NO DOWNSTREAM IMPACTS IDENTIFIED":

        assessment_valid = True

        feedback = (
            "Assessment completed. No retrieved resource "
            "was identified as genuinely impacted."
        )

    else:

        blocks = re.split(
            r"(?=DOCUMENT\s*:)",
            assessment,
            flags=re.IGNORECASE
        )

        # Ignore introductory text before the first DOCUMENT block.
        blocks = [
            block.strip()
            for block in blocks
            if block.strip()
            and re.search(
                r"DOCUMENT\s*:",
                block,
                re.IGNORECASE
            )
        ]

        assessment_valid = bool(blocks)

        feedback = []

        allowed_categories = {
            "artwork",
            "website content",
            "training material",
            "country submission"
        }

        allowed_priorities = {
            "high",
            "medium",
            "low"
        }

        retrieved_names = {
            document.get("name", "").strip().casefold()
            for document in retrieved_documents
        }

        for block in blocks:

            document_match = re.search(
                r"DOCUMENT\s*:\s*(.+)",
                block,
                re.IGNORECASE
            )

            category_match = re.search(
                r"CATEGORY\s*:\s*(.+)",
                block,
                re.IGNORECASE
            )

            impact_match = re.search(
                r"IMPACT\s*:\s*(.+)",
                block,
                re.IGNORECASE
            )

            priority_match = re.search(
                r"PRIORITY\s*:\s*(.+)",
                block,
                re.IGNORECASE
            )

            action_match = re.search(
                r"RECOMMENDED ACTION\s*:\s*(.+)",
                block,
                re.IGNORECASE
            )

            if not all([
                document_match,
                category_match,
                impact_match,
                priority_match,
                action_match
            ]):
                assessment_valid = False
                feedback.append(
                    "An assessment is missing required fields."
                )
                continue

            document_name = document_match.group(1).strip()
            category = category_match.group(1).strip().casefold()
            priority = priority_match.group(1).strip().casefold()

            if document_name.casefold() not in retrieved_names:
                assessment_valid = False
                feedback.append(
                    f"Unretrieved document referenced: {document_name}"
                )

            if category not in allowed_categories:
                assessment_valid = False
                feedback.append(
                    f"Invalid document category: {category}"
                )

            if priority not in allowed_priorities:
                assessment_valid = False
                feedback.append(
                    f"Invalid priority: {priority}"
                )

            if not impact_match.group(1).strip():
                assessment_valid = False
                feedback.append(
                    f"Missing impact explanation for {document_name}"
                )

            if not action_match.group(1).strip():
                assessment_valid = False
                feedback.append(
                    f"Missing recommended action for {document_name}"
                )

        feedback = (
            "; ".join(feedback)
            if feedback
            else "Assessment structure and document references validated."
        )

    if assessment_valid:
        decisions.append(
            "Impact assessment passed structural and evidence-reference validation"
        )
    else:
        decisions.append(
            "Impact assessment failed validation; human verification required"
        )

    return {
        "impact_assessment": assessment,
        "assessment_attempts": attempts,
        "assessment_valid": assessment_valid,
        "assessment_feedback": feedback,
        "human_review_required": True,
        "review_status": (
            "Pending human review"
            if assessment_valid
            else "Pending manual verification of assessment"
        ),
        "agent_steps": steps,
        "agent_decisions": decisions
    }

def generate_tasks_node(state):
    """
    Generate actionable change-impact tasks and validate
    them against the approved impact assessment.
    """

    import re

    assessment = state.get("impact_assessment", "")

    # Never generate tasks from an invalid assessment.
    if not state.get("assessment_valid", False):
        return {
            "task_list": (
                "TASK GENERATION BLOCKED: Impact assessment "
                "has not passed validation. Manual review required."
            ),
            "task_validation_passed": False,
            "task_validation_feedback": (
                "Task generation blocked because the impact "
                "assessment failed validation."
            ),
            "human_review_required": True,
            "review_status": "Pending impact assessment review"
        }

    # No downstream impacts means no tasks.
    if assessment.strip().upper() == "NO DOWNSTREAM IMPACTS IDENTIFIED":
        return {
            "task_list": "NO CHANGE-IMPACT TASKS REQUIRED",
            "task_validation_passed": True,
            "task_validation_feedback": (
                "No impacted downstream resources were identified."
            ),
            "human_review_required": True,
            "review_status": "Pending human review"
        }

    result = generate_task_list(
        state["detected_changes"],
        assessment
    )

    task_list = result.strip()

    steps = state.get("agent_steps", []).copy()
    steps.append("Change-impact tasks generated and validated")

    decisions = state.get("agent_decisions", []).copy()

    # Identify the resources approved in the impact assessment.
    approved_resources = {
        name.strip().casefold()
        for name in re.findall(
            r"DOCUMENT\s*:\s*(.+)",
            assessment,
            re.IGNORECASE
        )
    }

    # Separate individual tasks.
    task_blocks = re.split(
        r"(?=TASK ID\s*:)",
        task_list,
        flags=re.IGNORECASE
    )

    task_blocks = [
        block.strip()
        for block in task_blocks
        if block.strip()
    ]

    valid_tasks = []
    feedback = []

    if task_list.upper() == "NO CHANGE-IMPACT TASKS REQUIRED":

        validation_passed = True

    elif not task_blocks or not approved_resources:

        validation_passed = False
        feedback.append(
            "No valid tasks or approved impacted resources found."
        )

    else:

        validation_passed = True

        allowed_categories = {
            "artwork",
            "website content",
            "training material",
            "country submission"
        }

        allowed_priorities = {"high", "medium", "low"}

        allowed_statuses = {"pending"}

        for index, block in enumerate(task_blocks, start=1):

            patterns = {
                "task_id": r"TASK ID\s*:\s*(.+)",
                "resource": r"AFFECTED RESOURCE\s*:\s*(.+)",
                "category": r"CATEGORY\s*:\s*(.+)",
                "priority": r"PRIORITY\s*:\s*(.+)",
                "action": r"ACTION REQUIRED\s*:\s*(.+)",
                "reason": r"REASON\s*:\s*(.+)",
                "owner": r"SUGGESTED OWNER\s*:\s*(.+)",
                "status": r"STATUS\s*:\s*(.+)"
            }

            fields = {}

            for key, pattern in patterns.items():

                match = re.search(
                    pattern,
                    block,
                    re.IGNORECASE
                )

                fields[key] = (
                    match.group(1).strip()
                    if match else ""
                )

            expected_id = f"TASK-{index:03d}"

            if fields["task_id"] != expected_id:
                validation_passed = False
                feedback.append(
                    f"Invalid task ID: expected {expected_id}."
                )

            if fields["resource"].casefold() not in approved_resources:
                validation_passed = False
                feedback.append(
                    f"Task references an unapproved resource: "
                    f"{fields['resource']}"
                )

            if fields["category"].casefold() not in allowed_categories:
                validation_passed = False
                feedback.append(
                    f"Invalid category in {expected_id}."
                )

            if fields["priority"].casefold() not in allowed_priorities:
                validation_passed = False
                feedback.append(
                    f"Invalid priority in {expected_id}."
                )

            if fields["status"].casefold() not in allowed_statuses:
                validation_passed = False
                feedback.append(
                    f"Invalid status in {expected_id}."
                )

            for field in ("resource", "action", "reason", "owner"):
                if not fields[field]:
                    validation_passed = False
                    feedback.append(
                        f"Missing {field} in {expected_id}."
                    )

            valid_tasks.append(fields)

    validation_feedback = (
        "; ".join(feedback)
        if feedback
        else "Generated tasks passed structural and resource-reference validation."
    )

    decisions.append(
        "Task validation passed"
        if validation_passed
        else "Task validation failed; human verification required"
    )

    return {
        "task_list": task_list,
        "task_validation_passed": validation_passed,
        "task_validation_feedback": validation_feedback,
        "human_review_required": True,
        "review_status": (
            "Pending human review"
            if validation_passed
            else "Pending manual task verification"
        ),
        "agent_steps": steps,
        "agent_decisions": decisions
    }
