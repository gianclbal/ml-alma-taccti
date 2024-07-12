# custom_tokenizer.py

import re

def custom_sent_tokenizer(text):
    # Define custom sentence segmentation rules
    sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s"
    sentences = re.split(sentence_endings, text)
    return sentences
