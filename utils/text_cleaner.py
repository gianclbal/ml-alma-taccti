import re

def clean_text(text):
    # Remove dashes and new lines
    text = text.replace('-', '')  # Remove dashes
    text = text.replace('\n', ' ')  # Replace new lines with space
    
    # Define a regular expression pattern to keep only letters, numbers, and specified punctuation
    pattern = r'[^a-zA-Z0-9.,!?\'"()_:;\s]'
    cleaned_text = re.sub(pattern, '', text)
    
    # Remove extra spaces
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text
