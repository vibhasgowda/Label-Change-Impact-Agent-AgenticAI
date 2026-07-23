from tools.label_reader import read_document

file_path = input("Enter the full path of a PDF label: ").strip('"')

try:
    text = read_document(file_path)

    print("\n--- LABEL READER SUCCESS ---")
    print(f"Characters extracted: {len(text)}")
    print("\nFirst 1000 characters:\n")
    print(text[:1000])

except Exception as error:
    print(f"\nError: {error}")