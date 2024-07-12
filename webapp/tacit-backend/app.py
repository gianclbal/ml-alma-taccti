#!/usr/bin/python3
# render template
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json
import gensim
import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer
# ktrain to load model
import ktrain
# Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])
# Load the attainment model at the startup.
attainment_model = ktrain.load_predictor('./distillbert_model/')

# Initialize Count Vectorizer at the start.
vectorizer = CountVectorizer(analyzer='word',
                             # min_df=1,                      # minimum reqd occurences of a word
                             # stop_words='english',           # remove stop words
                             lowercase=True)                 # convert all words to lowercase
# token_pattern='[a-zA-Z0-9]{2,}')# num chars > 3
# max_features=50000,           # max number of uniq words

# Sentence to words


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# Lemmatization


# 'NOUN', 'ADJ', 'VERB', 'ADV'
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in [
                         '-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# Pre process text


def preprocess_text(lst):
    paragraphs = list(lst)
    # Remove emails if any
    lst = [re.sub(r'\S*@\S*\s?', '', item) for item in lst]
    # Remove newline characters.
    lst = [re.sub(r'\s+', ' ', item) for item in lst]
    # Remove single quotes if any.
    lst = [re.sub(r"\'", "", item) for item in lst]
    # Convert sentence to words
    text_words = list(sent_to_words(lst))
    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    text_lemmatized = lemmatization(text_words, allowed_postags=[
                                    'NOUN', 'VERB'])  # select noun and verb
    # Vectorize
    vec = vectorizer.fit(paragraphs)
    text_vectorized = vec.fit_transform(text_lemmatized)
    # Count the words
    sum_words = text_vectorized.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    return words_freq


# Define the app
app = Flask(__name__)
CORS(app, resources={
  r"/*": {
    "origins": "/*"
  }
})

# Route for index page of api.


@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting attainment for a single essay.


@app.route('/culturalcapitals', methods=['GET', 'POST'])
def culturalcapitals():
    response = {}
    # app data
    appData = json.loads(request.data)
    print("APP DATA START")
    print(appData)
    # Thematic Codes
    thematic_code = appData['thematicCode']['value']
    # Attainment
    attainment_present = 0
    attainment_missing = 0
    data_table = list()
    ## gian addition *******************************************
    appData["essays"] = []
    if (int(thematic_code) == 1):
        # count variable to be used as serial no in display.
        table_counter = 1
        for item in appData['csvData']:
            # data table entry
            entry = list()
            # add s no
            entry.append(str(table_counter))
            # add essay id
            entry.append(str(item['id']))
            # essay
            essay = str(item['essay'])
            appData["essays"].append(essay)
            # print(appData)
            # split the essay into sentences
            sentence_list = re.split('\.|\?|\!', essay)
            # get the predictions for the sentence list
            y_pred = attainment_model.predict(sentence_list)
            # checking if we found attainment
            if(int(max(y_pred)) > 0):
                attainment_present = attainment_present + 1
                entry.append('Yes')
                # if attainment present, then mark the sentences containing attainment.
                for i in range(len(y_pred)):
                    if int(y_pred[i]) == 1:
                        marked_str = '<mark>' + \
                            str(sentence_list[i]) + '</mark>'
                        essay = essay.replace(
                            str(sentence_list[i]), marked_str)
                entry.append(essay)
            else:
                attainment_missing = attainment_missing + 1
                entry.append('No')
                entry.append(essay)
            table_counter = table_counter + 1
            # add the entry to the data table list.
            data_table.append(entry)

    # Build Word Cloud
    paragraphs = appData['essays']
    print("PARAGRAPHS")
    print(paragraphs)
    # Pre process text to get the word frequency
    word_frequency = preprocess_text(paragraphs)
    total_words = 0
    word_cloud = list()
    # Create the required structure.
    for item in word_frequency:
        content = {}
        content["text"] = str(item[0])
        content["value"] = str(item[1])
        word_cloud.append(content)
        total_words = total_words + 1

    '''
  paragraphs = appData['paragraphs']
  # Pre process text to get the word frequency
  word_frequency = preprocess_text(paragraphs)
  total_words= 0
  word_cloud = list()
  # Create the required structure.
  for item in word_frequency:
    content = {}
    content["text"] = str(item[0])
    content["value"] = str(item[1])
    word_cloud.append(content)
    total_words = total_words + 1
  
  # Thematic codes
  thematic_code = appData['thematicCode']['value']
  # Attainment
  data_table = list()
  attainment_present = 0
  attainment_missing = 0
  # count variable to be used as serial no in display.
  counter = 1
  if (int(thematic_code) == 1):
    for item in appData['csvData']:
      entry = list()
      # add serial number
      entry.append(str(counter))
      sentence = str(item['sentence'])
      y_pred = attainment_model.predict(sentence)
      if (int(y_pred) == 1):
        attainment_present = attainment_present + 1
        entry.append('Yes')
      elif (int(y_pred) == 0):
        attainment_missing = attainment_missing + 1
        entry.append('No')
      counter = counter + 1
      # add sentence
      entry.append(sentence)
      data_table.append(entry)
  '''
    # Final response
    json_string = json.dumps({
        "word_cloud": word_cloud,
        "total_words": total_words,
        "data_table": data_table,
        "chart_counts": [attainment_present, attainment_missing]
    })
    return json_string


# Route for word cloud data processing
'''@app.route('/wordcloud', methods=['GET','POST'])
def word_cloud():
  response = {}
  # app data
  appData = json.loads(request.data)
  # Get paragraphs
  paragraphs = appData['paragraphs']
  # Pre process text to get the word frequency
  word_frequency = preprocess_text(paragraphs)
  total_words= 0
  word_cloud = list()
  # Create the required structure.
  for item in word_frequency:
    content = {}
    content["text"] = str(item[0])
    content["value"] = str(item[1])
    word_cloud.append(content)
    total_words = total_words + 1

  json_string = json.dumps({
    "word_cloud": word_cloud,
    "total_words": total_words
  })
  return json_string '''


# We only need this for local development.
if __name__ == '__main__':
    app.run(threaded=True)
