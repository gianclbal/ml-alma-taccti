import re

def clean_text(text):
    """
    Cleans input text by removing unwanted characters, handling dashes, 
    and converting text to lowercase.

    Parameters:
    - text (str): The input text to be cleaned.

    Returns:
    - str: The cleaned and standardized text.
    """
    if not isinstance(text, str):  # Ensure input is a string
        return ""

    # Replace specific patterns
    text = text.replace('-', '')  # Remove dashes
    text = text.replace('\n', ' ')  # Replace new lines with space
    text = text.replace('/-/', '...')  # Convert custom dash separator to ellipsis

    # Define a regex pattern to keep only letters, numbers, and specified punctuation
    pattern = r"[^a-zA-Z0-9.,!?\'\"()_:;\s]"
    cleaned_text = re.sub(pattern, '', text)

    # Convert to lowercase
    cleaned_text = cleaned_text.lower()

    # Remove extra spaces
    cleaned_text = ' '.join(cleaned_text.split())

    return cleaned_text