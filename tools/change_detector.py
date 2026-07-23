import re
import difflib
from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.1",
    temperature=0
)


# Key safety/regulatory sections we care about
SECTION_PATTERNS = {
    "BOXED WARNING": [
        "boxed warning",
        "warning:"
    ],
    "CONTRAINDICATIONS": [
        "contraindications"
    ],
    "WARNINGS AND PRECAUTIONS": [
        "warnings and precautions"
    ],
    "DOSAGE AND ADMINISTRATION": [
        "dosage and administration"
    ],
    "ADVERSE REACTIONS": [
        "adverse reactions"
    ],
    "DRUG INTERACTIONS": [
        "drug interactions"
    ]
}


def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text to reduce formatting noise.
    """

    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def find_section_start(text: str, possible_titles: list):
    """
    Find the position of a pharmaceutical section heading.
    """

    for title in possible_titles:

        pattern = rf"(?im)^\s*(?:\d+(?:\.\d+)*)?\s*{re.escape(title)}\s*$"

        match = re.search(
            pattern,
            text
        )

        if match:
            return match.start()

    return None


def extract_sections(text: str) -> dict:
    """
    Extract important pharmaceutical safety sections.
    """

    text = normalize_text(text)

    section_positions = []

    for section_name, titles in SECTION_PATTERNS.items():

        position = find_section_start(
            text,
            titles
        )

        if position is not None:

            section_positions.append(
                (
                    position,
                    section_name
                )
            )

    section_positions.sort()

    sections = {}

    for index, (start, section_name) in enumerate(section_positions):

        if index + 1 < len(section_positions):

            end = section_positions[index + 1][0]

        else:

            end = len(text)

        sections[section_name] = text[
            start:end
        ].strip()

    return sections


def section_similarity(
    old_section: str,
    new_section: str
) -> float:

    return difflib.SequenceMatcher(
        None,
        old_section,
        new_section
    ).ratio()


def extract_changed_sections(
    old_text: str,
    new_text: str
) -> list:
    """
    Compare safety sections and return only sections
    that actually changed.
    """

    old_sections = extract_sections(
        old_text
    )

    new_sections = extract_sections(
        new_text
    )

    changed_sections = []

    all_sections = set(
        old_sections.keys()
    ) | set(
        new_sections.keys()
    )

    for section_name in all_sections:

        old_content = old_sections.get(
            section_name,
            ""
        )

        new_content = new_sections.get(
            section_name,
            ""
        )

        # Section added
        if not old_content and new_content:

            changed_sections.append(
                {
                    "section": section_name,
                    "status": "Added",
                    "old_content": "",
                    "new_content": new_content,
                    "similarity": 0
                }
            )

            continue

        # Section removed
        if old_content and not new_content:

            changed_sections.append(
                {
                    "section": section_name,
                    "status": "Removed",
                    "old_content": old_content,
                    "new_content": "",
                    "similarity": 0
                }
            )

            continue

        similarity = section_similarity(
            old_content,
            new_content
        )

        # Ignore nearly identical sections
        if similarity >= 0.995:
            continue

        changed_sections.append(
            {
                "section": section_name,
                "status": "Modified",
                "old_content": old_content,
                "new_content": new_content,
                "similarity": round(
                    similarity * 100,
                    2
                )
            }
        )

    return changed_sections


def analyze_section_change(
    change: dict
) -> str:
    """
    Use Llama only to interpret ONE already-detected
    section change.
    """

    section = change["section"]

    old_content = change[
        "old_content"
    ][:7000]

    new_content = change[
        "new_content"
    ][:7000]

    prompt = f"""
You are a pharmaceutical regulatory change-impact analyst.

Python has already determined that the pharmaceutical section
below has changed.

Your task is ONLY to explain the meaningful regulatory or
safety change between the OLD and NEW versions.

SECTION:
{section}

OLD VERSION:
{old_content}

NEW VERSION:
{new_content}

IMPORTANT RULES:

1. Compare OLD versus NEW directly.
2. Do not summarize the complete sections.
3. Ignore formatting, spacing, numbering and typographical changes.
4. Report only genuine medical, safety or regulatory changes.
5. Do not invent information.
6. Do not offer further assistance.
7. If there is no meaningful regulatory or safety change,
   respond exactly with:

NO MEANINGFUL CHANGE

Otherwise respond EXACTLY using:

CHANGE TYPE: Added / Modified / Removed

SECTION: {section}

OLD INFORMATION:
[What the old version stated]

NEW INFORMATION:
[What changed in the new version]

SAFETY SIGNIFICANCE:
[High / Medium / Low] - [Reason]

POTENTIAL DOWNSTREAM IMPACT:
[List only relevant categories from:
Artwork, Website Content, Training Material, Country Submission]
"""

    response = llm.invoke(
        prompt
    )

    return response.content.strip()


def detect_changes(
    old_text: str,
    new_text: str
) -> dict:
    """
    Main change detection function.
    """

    if not old_text or not new_text:

        raise ValueError(
            "Both old and new label text are required."
        )

    overall_similarity = (
        difflib.SequenceMatcher(
            None,
            old_text,
            new_text
        ).ratio()
    )

    similarity_percentage = round(
        overall_similarity * 100,
        2
    )

    changed_sections = extract_changed_sections(
        old_text,
        new_text
    )

    meaningful_results = []

    for change in changed_sections:

        analysis = analyze_section_change(
            change
        )

        if (
            "NO MEANINGFUL CHANGE"
            not in analysis.upper()
        ):

            meaningful_results.append(
                analysis
            )

    if meaningful_results:

        final_analysis = (
            "\n\n"
            + "\n\n---\n\n".join(
                meaningful_results
            )
        )

    else:

        final_analysis = (
            "NO MEANINGFUL SAFETY CHANGES DETECTED"
        )

    return {
        "similarity_percentage":
            similarity_percentage,

        "change_analysis":
            final_analysis,

        "changed_sections":
            changed_sections
    }