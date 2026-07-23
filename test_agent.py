from agent.graph import label_change_agent


print("LABEL CHANGE IMPACT AGENT TEST STARTED")

try:
    old_path = input(
        "Enter the full path of the OLD label PDF: "
    ).strip('"')

    new_path = input(
        "Enter the full path of the NEW label PDF: "
    ).strip('"')

    print("\nRunning complete Agentic AI workflow...")
    print("This may take a few minutes. Please wait.\n")

    initial_state = {
        "old_label_path": old_path,
        "new_label_path": new_path,
        "agent_steps": []
    }

    result = label_change_agent.invoke(
        initial_state
    )

    print("=" * 60)
    print("AGENT WORKFLOW COMPLETED")
    print("=" * 60)

    print("\nAGENT EXECUTION STEPS:")

    for number, step in enumerate(
        result["agent_steps"],
        start=1
    ):
        print(f"{number}. {step}")

    print("\n" + "=" * 60)
    print("DETECTED LABEL CHANGES")
    print("=" * 60)

    print(result["detected_changes"])

    print("\n" + "=" * 60)
    print("DOWNSTREAM IMPACT ASSESSMENT")
    print("=" * 60)

    print(result["impact_assessment"])

    print("\n" + "=" * 60)
    print("FINAL CHANGE-IMPACT TASK LIST")
    print("=" * 60)

    print(result["task_list"])

    print("\n" + "=" * 60)
    print("END-TO-END AGENT TEST SUCCESSFUL")
    print("=" * 60)

except Exception as error:
    print(f"\nAGENT ERROR: {error}")