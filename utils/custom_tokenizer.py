import re

def custom_sent_tokenizer(text, include_question_mark=False):
    """
    Tokenizes text into sentences, ensuring proper splitting for standard sentence endings 
    as well as custom markers (/%/ and /-/).

    Parameters:
    - text (str): The input text to be tokenized.
    - include_question_mark (bool): If True, treats "?" as a sentence delimiter. Default is False.

    Returns:
    - list: A list of tokenized sentences.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    if not include_question_mark:
        print("Note: (?) is NOT treated as a delimiter because it often appears in rhetorical questions or as part of the essay prompt.")

    # Step 1: First split by custom markers "/%/" and "/-/" (properly escaped)
    temp_sentences = re.split(r"\/%\/|\/-\/", text)  # Ensure '/' is escaped properly

    # Step 2: Define standard sentence-ending regex (conditionally including "?")
    if include_question_mark:
        sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s"
    else:
        sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\!)\s"  # Excludes "?"

    # Step 3: Further split sentences using punctuation-based sentence breaking
    sentences = []
    for segment in temp_sentences:
        sentences.extend(re.split(sentence_endings, segment))

    # Step 4: Clean up sentences (remove empty strings and extra spaces)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences