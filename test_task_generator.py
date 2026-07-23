from tools.task_generator import generate_task_list


print("TASK GENERATOR TEST STARTED")

try:
    detected_change = """
A new requirement has been added to monitor renal function
every 6 months in patients at increased risk of renal impairment
while using Metformin Hydrochloride Tablets.
"""

    impact_assessment = """
DOCUMENT: metformin_safety_training
CATEGORY: Training Material
IMPACT: The new renal monitoring requirement must be included
in the safety training material.
PRIORITY: High
RECOMMENDED ACTION: Review and update the training material
to reflect the new monitoring requirement.

---

DOCUMENT: metformin_regulatory_submission
CATEGORY: Country Submission
IMPACT: The new patient-monitoring requirement may require
regulatory assessment in applicable countries.
PRIORITY: Medium
RECOMMENDED ACTION: Evaluate whether a variation, notification,
or updated submission is required.

---

DOCUMENT: metformin_product_page
CATEGORY: Website Content
IMPACT: The website contains renal impairment information and
should be reviewed for alignment with the updated label.
PRIORITY: Low
RECOMMENDED ACTION: Review and update the website content
if required.

---

DOCUMENT: metformin_packaging
CATEGORY: Artwork
IMPACT: The artwork does not directly contain the new monitoring
requirement.
PRIORITY: Low
RECOMMENDED ACTION: No action required.
"""

    print(
        "\nGenerating Change-Impact Task List "
        "with Llama 3.1... Please wait.\n"
    )

    result = generate_task_list(
        detected_change,
        impact_assessment
    )

    print("=" * 60)
    print("CHANGE-IMPACT TASK LIST")
    print("=" * 60)

    print(result)

    print("\n" + "=" * 60)
    print("TASK GENERATION COMPLETED")
    print("=" * 60)

except Exception as error:
    print(f"\nError: {error}")