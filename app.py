
import os
import uuid
import tempfile

import streamlit as st
from langgraph.types import Command

from agent.graph import label_change_agent


st.set_page_config(
    page_title="Pharmaceutical Label Change Impact Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Pharmaceutical Label Change Impact Agent")
st.caption(
    "Agentic AI for pharmaceutical label change detection, "
    "downstream impact assessment, and actionable task generation."
)

st.markdown("---")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "graph_config" not in st.session_state:
    st.session_state.graph_config = None

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.header("1. Upload Pharmaceutical Labels")

col1, col2 = st.columns(2)

with col1:
    old_file = st.file_uploader(
        "Upload Old Label (PDF)",
        type=["pdf"],
        key="old_label"
    )

with col2:
    new_file = st.file_uploader(
        "Upload Updated Label (PDF)",
        type=["pdf"],
        key="new_label"
    )


# --------------------------------------------------
# RUN AGENT
# --------------------------------------------------

if st.button(
    "Run Label Change Impact Analysis",
    type="primary",
    disabled=not (old_file and new_file)
):

    temp_paths = []

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as old_temp:
            old_temp.write(old_file.getvalue())
            old_path = old_temp.name
            temp_paths.append(old_path)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as new_temp:
            new_temp.write(new_file.getvalue())
            new_path = new_temp.name
            temp_paths.append(new_path)

        thread_id = str(uuid.uuid4())

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        initial_state = {
            "old_label_path": old_path,
            "new_label_path": new_path,
            "agent_steps": [],
            "agent_decisions": [],
            "retrieval_attempts": 0,
            "max_retrieval_attempts": 2,
            "assessment_attempts": 0
        }

        with st.spinner(
            "Agent is analyzing the pharmaceutical labels..."
        ):
            result = label_change_agent.invoke(
                initial_state,
                config=config
            )

        st.session_state.result = result
        st.session_state.graph_config = config
        st.session_state.analysis_started = True

        st.success("Analysis workflow executed.")

    except Exception as e:
        st.error(f"Workflow error: {str(e)}")

    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

result = st.session_state.result

if result:

    st.markdown("---")
    st.header("2. Agent Workflow Execution")

    steps = result.get("agent_steps", [])

    if steps:
        for index, step in enumerate(steps, start=1):
            st.write(f"**Step {index}:** {step}")
    else:
        st.info("No workflow steps recorded.")

    st.markdown("---")
    st.header("3. Agent Decisions")

    decisions = result.get("agent_decisions", [])

    if decisions:
        for decision in decisions:
            st.write(f"• {decision}")
    else:
        st.info("No agent decisions recorded.")

    st.markdown("---")
    st.header("4. Analysis Metrics")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Similarity",
        f"{result.get('similarity_percentage', 0):.2f}%"
    )

    metric2.metric(
        "Meaningful Change",
        "Yes" if result.get(
            "meaningful_change_detected", False
        ) else "No"
    )

    metric3.metric(
        "Assessment Valid",
        "Yes" if result.get("assessment_valid") else "No"
    )

    st.markdown("---")
    st.header("5. Detected Label Changes")

    st.text(result.get(
        "detected_changes",
        "No detected changes available."
    ))

    st.markdown("---")
    st.header("6. Retrieved Downstream Documents")

    documents = result.get("retrieved_documents", [])

    if documents:
        for doc in documents:
            with st.expander(
                doc.get("document_name", "Retrieved document")
            ):
                st.write(
                    "**Category:**",
                    doc.get("category", "N/A")
                )
                st.write(
                    "**Similarity Score:**",
                    doc.get("similarity_score", "N/A")
                )
                st.write(
                    doc.get("content", "")
                )
    else:
        st.info("No downstream documents retrieved.")

    st.markdown("---")
    st.header("7. Impact Assessment")

    st.text(result.get(
        "impact_assessment",
        "No impact assessment available."
    ))

    if result.get("assessment_feedback"):
        st.info(result["assessment_feedback"])

    st.markdown("---")
    st.header("8. Generated Change-Impact Tasks")

    task_list = result.get("task_list", "")

    if task_list:
        st.text(task_list)
    else:
        st.info("No tasks generated.")

    if result.get("task_validation_feedback"):
        st.caption(result["task_validation_feedback"])

    if result.get("task_validation_passed"):
        st.success("Task validation passed.")
    elif task_list:
        st.warning("Task validation requires attention.")

    # --------------------------------------------------
    # HUMAN APPROVAL
    # --------------------------------------------------

    st.markdown("---")
    st.header("9. Human Review and Approval")

    is_paused = bool(result.get("__interrupt__"))

    if is_paused:

        st.warning(
            "The agent has paused execution and is waiting "
            "for a human decision."
        )

        st.write(
            "Review the impact assessment and generated tasks "
            "before making a decision."
        )

        approve_col, reject_col = st.columns(2)

        with approve_col:
            if st.button(
                "Approve Tasks",
                type="primary",
                use_container_width=True
            ):

                try:
                    with st.spinner(
                        "Resuming workflow with human approval..."
                    ):
                        resumed_result = label_change_agent.invoke(
                            Command(
                                resume={"approved": True}
                            ),
                            config=st.session_state.graph_config
                        )

                    st.session_state.result = resumed_result
                    st.rerun()

                except Exception as e:
                    st.error(f"Approval error: {str(e)}")

        with reject_col:
            if st.button(
                "Reject Tasks",
                type="secondary",
                use_container_width=True
            ):

                try:
                    with st.spinner(
                        "Resuming workflow with rejection..."
                    ):
                        resumed_result = label_change_agent.invoke(
                            Command(
                                resume={"approved": False}
                            ),
                            config=st.session_state.graph_config
                        )

                    st.session_state.result = resumed_result
                    st.rerun()

                except Exception as e:
                    st.error(f"Rejection error: {str(e)}")

    else:

        approval = result.get("human_approved")
        review_status = result.get(
            "review_status",
            "No human decision recorded"
        )

        st.write("**Review status:**", review_status)

        if approval is True:
            st.success("Tasks approved by the human reviewer.")

        elif approval is False:
            st.error("Tasks rejected by the human reviewer.")

        elif result.get("task_list"):
            st.info(
                "No pending approval interrupt was returned. "
                "Check the workflow status."
            )

    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------

    st.markdown("---")
    st.header("10. Download Analysis Report")

    report = f"""
PHARMACEUTICAL LABEL CHANGE IMPACT ANALYSIS
============================================

DETECTED CHANGES
----------------
{result.get('detected_changes', '')}

IMPACT ASSESSMENT
-----------------
{result.get('impact_assessment', '')}

GENERATED TASKS
---------------
{result.get('task_list', '')}

HUMAN REVIEW STATUS
-------------------
{result.get('review_status', 'Pending')}

HUMAN APPROVAL
--------------
{result.get('human_approved', 'Pending')}

WORKFLOW STEPS
--------------
{chr(10).join(result.get('agent_steps', []))}

AGENT DECISIONS
---------------
{chr(10).join(result.get('agent_decisions', []))}
"""

    st.download_button(
        label="Download Analysis Report",
        data=report,
        file_name="label_change_impact_report.txt",
        mime="text/plain"
    )


else:

    st.info(
        "Upload the old and updated pharmaceutical label PDFs "
        "to begin the analysis."
    )


st.markdown("---")
st.caption(
    "Research prototype. Generated assessments and tasks "
    "require qualified human review. Synthetic test documents "
    "must not be treated as approved regulatory or clinical material."
)
