import pandas as pd
import numpy as np
import random
import re
import warnings
import stanza
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from collections import Counter
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer

# Set random seed
random.seed(18)
seed = 18

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stanza.download('en')  # download English model
nlp = stanza.Pipeline('en')  # initialize English neural pipeline

def get_ner(text):
    ner_list = []
    doc = nlp(text)
    for sentence in doc.sentences:
        for entity in sentence.ents:
            if entity.type == 'PERSON':
                ner_list.append(entity.text)
    return ner_list

def named_entity_present(sentence):
    ner_list = get_ner(sentence)
    return int(len(ner_list) > 0)

def word_similarity(tokens, syns, field):    
    score_threshold = 0.5 if field in ['engineering', 'technology'] else 0.2
    sim_words = 0
    for token in tokens:
        if token not in stop_words:
            try:
                syns_word = wordnet.synsets(token) 
                score = syns_word[0].path_similarity(syns[0])
                if score >= score_threshold:
                    sim_words += 1
            except IndexError: 
                pass
    return sim_words

def check_medical_words(tokens, medical_list):
    return int(any(token not in stop_words and token in medical_list for token in tokens))

def get_sentiment(sentence):
    sentiments = TextBlob(sentence).sentiment
    return sentiments.polarity, sentiments.subjectivity

def count_pos_tags(tokens):
    token_pos = pos_tag(tokens)
    count = Counter(tag for word, tag in token_pos)
    return (count['UH'], count['NN'] + count['NNS'] + count['NNP'] + count['NNPS'],
            count['RB'] + count['RBS'] + count['RBR'], 
            count['VB'] + count['VBD'] + count['VBG'] + count['VBN'], 
            count['DT'], count['PRP'], 
            count['JJ'] + count['JJR'] + count['JJS'], count['IN'])

def pos_tag_extraction(dataframe, field, func, column_names):
    return pd.concat((dataframe, dataframe[field].apply(lambda cell: pd.Series(func(cell), index=column_names))), axis=1)

def load_medical_list(file_path):
    medical_list = []
    with open(file_path, 'r') as medical_fields:
        for line in medical_fields.readlines():
            special_field = re.sub(r"\W", " ", line.rstrip('\n'))
            medical_list += special_field.split()
    return list(set(medical_list))

def feature_engineering(dataset, medical_list):
    dataset['tokens'] = dataset['sentence'].apply(word_tokenize)
    
    # Similarity features
    syns_bio = wordnet.synsets(lemmatizer.lemmatize("biology"))
    syns_chem = wordnet.synsets(lemmatizer.lemmatize("chemistry"))
    syns_phy = wordnet.synsets(lemmatizer.lemmatize("physics"))
    syns_maths = wordnet.synsets(lemmatizer.lemmatize("mathematics"))
    syns_tech = wordnet.synsets(lemmatizer.lemmatize("technology"))
    syns_eng = wordnet.synsets(lemmatizer.lemmatize("engineering"))

    dataset['bio_sim_words'] = dataset['tokens'].apply(word_similarity, args=(syns_bio, 'biology'))
    dataset['chem_sim_words'] = dataset['tokens'].apply(word_similarity, args=(syns_chem, 'chemistry'))
    dataset['phy_sim_words'] = dataset['tokens'].apply(word_similarity, args=(syns_phy, 'physics'))
    dataset['math_sim_words'] = dataset['tokens'].apply(word_similarity, args=(syns_maths, 'mathematics'))
    dataset['tech_sim_words'] = dataset['tokens'].apply(word_similarity, args=(syns_tech, 'technology'))
    dataset['eng_sim_words'] = dataset['tokens'].apply(word_similarity, args=(syns_eng, 'engineering'))

    dataset['medical_terms'] = dataset['tokens'].apply(lambda tokens: check_medical_words(tokens, medical_list))
    dataset['polarity'], dataset['subjectivity'] = zip(*dataset['sentence'].apply(get_sentiment))
    dataset['ner'] = dataset['sentence'].apply(named_entity_present)
    dataset = pos_tag_extraction(dataset, 'tokens', count_pos_tags, ['interjections', 'nouns', 'adverb', 'verb', 'determiner', 'pronoun', 'adjective', 'preposition'])
    
    return dataset.drop(columns='label'), dataset['label']

def create_unigram_features(dataset):
    

    unigram_matrix = unigram_vect.fit_transform(dataset['sentence'])
    unigrams = pd.DataFrame(unigram_matrix.toarray())
    unigrams = unigrams.reset_index()
    unigrams = unigrams.iloc[:,1:]
    return pd.concat([dataset.reset_index(drop=True), unigrams.reset_index(drop=True)], axis=1)

# Example usage: Load medical specialties and prepare features for a DataFrame
def prepare_features(df, medical_file_path='/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/data/features/medical_specialities.txt'):
    print("Preparing features...")
    medical_list = load_medical_list(medical_file_path)
    unigram_vect = CountVectorizer(ngram_range=(1, 1), min_df=2, stop_words='english')
    
    X, y = feature_engineering(df, medical_list)
    X = create_unigram_features(X)


    return X, y
