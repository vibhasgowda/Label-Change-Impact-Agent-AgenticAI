print("TEST SCRIPT STARTED")

from tools.label_reader import read_document
from tools.change_detector import detect_changes


old_path = input(
    "Enter the full path of the OLD label PDF: "
).strip('"')

new_path = input(
    "Enter the full path of the NEW label PDF: "
).strip('"')


try:

    print("\nReading labels...")

    old_text = read_document(old_path)
    new_text = read_document(new_path)

    print("Labels read successfully.")

    print(
        "Analyzing actual label differences "
        "with Llama 3.1... Please wait.\n"
    )

    result = detect_changes(
        old_text,
        new_text
    )

    print("=" * 60)
    print("LABEL CHANGE ANALYSIS")
    print("=" * 60)

    print(
        f"\nOverall Similarity: "
        f"{result['similarity_percentage']}%"
    )

    print("\nDetected Safety Changes:\n")

    print(
        result["change_analysis"]
    )

    print("\n" + "=" * 60)
    print("CHANGE DETECTION COMPLETED")
    print("=" * 60)


except Exception as error:

    print(
        f"\nError: {error}"
    )