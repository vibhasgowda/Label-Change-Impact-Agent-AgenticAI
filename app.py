import os
import tempfile

import streamlit as st

from agent.graph import label_change_agent


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Label Change Impact Agent",
    page_icon="🔬",
    layout="wide"
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Label Change Impact Agent")

st.markdown(
    """
    **Agentic AI for Pharmaceutical Regulatory Operations**

    Upload an existing pharmaceutical product label and its updated
    version. The AI agent analyzes the label change, searches downstream
    content, assesses regulatory impact, and generates an actionable
    **Change-Impact Task List**.
    """
)

st.divider()


# ---------------------------------------------------------
# Upload Section
# ---------------------------------------------------------

st.subheader("Upload Product Labels")

col1, col2 = st.columns(2)

with col1:

    old_label = st.file_uploader(
        "Old Product Label",
        type=["pdf"],
        key="old_label"
    )

with col2:

    new_label = st.file_uploader(
        "Updated Product Label",
        type=["pdf"],
        key="new_label"
    )


# ---------------------------------------------------------
# Run Analysis
# ---------------------------------------------------------

if st.button(
    "Run Impact Analysis",
    type="primary",
    use_container_width=True
):

    if old_label is None or new_label is None:

        st.error(
            "Please upload both the old and updated product labels."
        )

    else:

        old_temp_path = None
        new_temp_path = None

        try:

            # ---------------------------------------------
            # Save uploaded PDFs temporarily
            # ---------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as old_temp:

                old_temp.write(
                    old_label.getvalue()
                )

                old_temp_path = old_temp.name


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as new_temp:

                new_temp.write(
                    new_label.getvalue()
                )

                new_temp_path = new_temp.name


            # ---------------------------------------------
            # Run Complete LangGraph Agent
            # ---------------------------------------------

            with st.spinner(
                "Agent is analyzing the label change. "
                "This may take a few minutes..."
            ):

                initial_state = {

                    "old_label_path":
                        old_temp_path,

                    "new_label_path":
                        new_temp_path,

                    "agent_steps":
                        []
                }

                result = label_change_agent.invoke(
                    initial_state
                )


            st.success(
                "Label change impact analysis completed."
            )


            # ---------------------------------------------
            # Agent Execution
            # ---------------------------------------------

            st.subheader(
                "Agent Workflow"
            )

            for step in result.get(
                "agent_steps",
                []
            ):

                st.success(
                    f"✓ {step}"
                )


            # ---------------------------------------------
            # Summary Metrics
            # ---------------------------------------------

            st.subheader(
                "Analysis Overview"
            )

            metric1, metric2 = st.columns(2)

            with metric1:

                st.metric(
                    "Label Similarity",
                    f"{result.get('similarity_percentage', 0)}%"
                )

            with metric2:

                retrieved_documents = result.get(
                    "retrieved_documents",
                    []
                )

                st.metric(
                    "Resources Retrieved",
                    len(retrieved_documents)
                )


            # ---------------------------------------------
            # Detected Changes
            # ---------------------------------------------

            st.subheader(
                "Detected Label Changes"
            )

            with st.expander(
                "View detected safety and regulatory changes",
                expanded=True
            ):

                st.text(
                    result.get(
                        "detected_changes",
                        "No changes available."
                    )
                )


            # ---------------------------------------------
            # Retrieved Resources
            # ---------------------------------------------

            st.subheader(
                "Retrieved Downstream Resources"
            )

            retrieved_documents = result.get(
                "retrieved_documents",
                []
            )

            if retrieved_documents:

                for document in retrieved_documents:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"**{document['name']}**"
                        )

                        st.write(
                            f"Category: "
                            f"{document['category']}"
                        )

                        st.write(
                            "Semantic Similarity: "
                            f"{document.get('similarity_score', 0):.4f}"
                        )

            else:

                st.info(
                    "No downstream resources were retrieved."
                )


            # ---------------------------------------------
            # Impact Assessment
            # ---------------------------------------------

            st.subheader(
                "Downstream Impact Assessment"
            )

            with st.expander(
                "View detailed impact assessment",
                expanded=True
            ):

                st.text(
                    result.get(
                        "impact_assessment",
                        "No impact assessment available."
                    )
                )


            # ---------------------------------------------
            # Final Change-Impact Task List
            # ---------------------------------------------

            st.subheader(
                "Change-Impact Task List"
            )

            st.info(
                "The following actionable tasks were generated "
                "by the agent based on the detected label changes "
                "and downstream impact assessment."
            )

            st.text(
                result.get(
                    "task_list",
                    "No tasks generated."
                )
            )


            # ---------------------------------------------
            # Download Report
            # ---------------------------------------------

            report = f"""
LABEL CHANGE IMPACT REPORT

============================================================
DETECTED LABEL CHANGES
============================================================

{result.get("detected_changes", "")}


============================================================
DOWNSTREAM IMPACT ASSESSMENT
============================================================

{result.get("impact_assessment", "")}


============================================================
CHANGE-IMPACT TASK LIST
============================================================

{result.get("task_list", "")}
"""


            st.download_button(
                label="Download Impact Report",
                data=report,
                file_name="label_change_impact_report.txt",
                mime="text/plain",
                use_container_width=True
            )


        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )


        finally:

            # ---------------------------------------------
            # Delete temporary uploaded files
            # ---------------------------------------------

            if (
                old_temp_path
                and os.path.exists(
                    old_temp_path
                )
            ):

                os.remove(
                    old_temp_path
                )


            if (
                new_temp_path
                and os.path.exists(
                    new_temp_path
                )
            ):

                os.remove(
                    new_temp_path
                )