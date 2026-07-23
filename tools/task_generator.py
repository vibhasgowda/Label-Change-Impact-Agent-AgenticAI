from langchain_ollama import ChatOllama


# Local Llama 3.1 model through Ollama
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)


def generate_task_list(
    detected_changes: str,
    impact_assessment: str
) -> str:
    """
    Generate the final Change-Impact Task List.

    This is the primary deliverable of the
    Label Change Impact Agent.
    """

    if not detected_changes:
        raise ValueError(
            "Detected label changes are required."
        )

    if not impact_assessment:
        raise ValueError(
            "Impact assessment is required."
        )

    prompt = f"""
You are the Task Generation Tool inside an Agentic AI system
for pharmaceutical regulatory operations.

A pharmaceutical label change has been analyzed and its downstream
impact has already been assessed.

Your task is to convert the assessment into a clear and actionable
CHANGE-IMPACT TASK LIST.

DETECTED LABEL CHANGES:

{detected_changes}

DOWNSTREAM IMPACT ASSESSMENT:

{impact_assessment}

Create tasks ONLY for resources that require an action.

Do NOT create a task when the recommended action says:
- No action required
- No update required
- Not impacted

For every required task, return EXACTLY this structure:

TASK ID: TASK-001

AFFECTED RESOURCE:
[Document name]

CATEGORY:
[Artwork / Website Content / Training Material / Country Submission]

PRIORITY:
[High / Medium / Low]

ACTION REQUIRED:
[Specific action that should be completed]

REASON:
[Why the task is required based on the detected label change]

SUGGESTED OWNER:
[Regulatory Affairs / Medical Affairs / Training Team /
Digital Content Team / Artwork Team]

STATUS:
Pending

---

Number the task IDs sequentially:
TASK-001, TASK-002, TASK-003 and so on.

IMPORTANT RULES:

1. Generate only actionable tasks.
2. Do not invent resources.
3. Use only resources identified in the impact assessment.
4. Do not generate tasks for resources requiring no action.
5. Keep actions specific and practical.
6. Do not add conversational comments.
7. The final output must be a task list, not a general summary.

If no downstream action is required, return exactly:

NO CHANGE-IMPACT TASKS REQUIRED
"""

    # Generate task list using Llama 3.1
    response = llm.invoke(prompt)

    result = response.content.strip()

    # Remove unnecessary introductory text before the first task
    if "TASK-001" in result:
        result = (
            "TASK-001"
            + result.split("TASK-001", 1)[1]
        )

    return result