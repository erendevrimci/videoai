import pandas as pd
import spacy

# Load the small English pipeline from spaCy
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text, max_keywords=8):
    """
    Extract a set of noun/proper-noun keywords from the text.
    Returns a comma-separated string of keywords.
    
    :param text: The input text from which to extract keywords
    :param max_keywords: Maximum number of keywords to return (optional)
    :return: Comma-separated string of keywords
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Process the text with spaCy
    doc = nlp(text)

    # Collect nouns and proper nouns as candidate keywords
    keywords = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            # Clean the token text (lowercase, strip punctuation, etc.)
            clean_token = token.lemma_.lower().strip()
            # Avoid duplicates
            if clean_token not in keywords:
                keywords.append(clean_token)

    # (Optional) Limit the number of keywords
    keywords = keywords[:max_keywords]

    return ", ".join(keywords)

def main():
    # Path to your CSV file
    input_csv_path = "video\\video-catalog.csv"
    output_csv_path = "video\\video-catalog-labeled.csv"
    
    # Read the CSV (assuming tab-separated; adjust if comma-separated)
    df = pd.read_csv(input_csv_path, sep=",")

    # Ensure the 'labels' column exists, or create it if missing
    if 'labels' not in df.columns:
        df['labels'] = ""

    # Extract keywords for each row and update 'labels'
    for idx, row in df.iterrows():
        prompt_text = row.get("prompt", "")
        extracted_labels = extract_keywords(prompt_text)
        df.at[idx, "labels"] = extracted_labels

    # Save the updated CSV
    df.to_csv(output_csv_path, sep=",", index=False)
    print(f"Labels extracted and saved to {output_csv_path}")

if __name__ == "__main__":
    main()
