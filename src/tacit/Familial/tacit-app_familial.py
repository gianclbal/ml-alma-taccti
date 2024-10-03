'''
INSTRUCTIONS:
1. Go to folder where script is in the terminal.
2. In terminal, run the following:
    python3 tacit-app.py <input_path_file>
    ex. python3 tacit-app.py ./sample_data/

OUTPUT:
The script will create a folder called "output" in the same folder as the script.

PURPOSE:
The purpose of tacit-app.py is to create a script that can mass-label and annotate essays candidatory for csv.
This script takes an input of an anonymized csv file and will output the same anonymized file but now in xlsx format and with 3 added columns:
    * Essay ID
    * CCT Present
    * Annotated Essay (replaces "Essay")
        * Includes rich writing. Sentences containing CCT are marked in bold and red.
This script allows human annotators to quickly scan a csv file and confirm the annotations of TACIT.

AUTHORS:
Gian Carlo L. Baldonado, Shailesh...

DATE:
March 17, 2023
'''



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
    paragraphs = [str(x or "") for x in paragraphs]
    # Remove emails if any
    paragraphs = [re.sub(r'\S*@\S*\s?', '', item) for item in paragraphs]
    # Remove newline characters.
    paragraphs = [re.sub(r'\s+', ' ', item) for item in paragraphs]
    # Remove single quotes if any.
    paragraphs = [re.sub(r"\'", "", item) for item in paragraphs]
    # Convert sentence to words
    text_words = list(sent_to_words(paragraphs))
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

# Get the string between two characters/words
def getSubstring(ch1,ch2,s):
    m = re.findall(ch1+'(.+?)'+ch2, s)
    
    return 

# Create the output directory and subdirectories
def generate_folder(dir_path, theme):
    """
    Creates hierarchy of directories for the output.
    Follows the structure: [Root Output] > [Semester Year] > [Courese Name and ID] > [Section] > [*.csv]
    """
    folder_dic = {"season": "", "year": "", "course_name": "", "section": ""}
   
    try:
        season = re.search("Fall|Spring|FALL|SPRING", dir_path).group()
        # print(season)
        year = re.search("20\d+", dir_path).group()
        # print(year)
        course_name = re.search("((PHYS|CHEM|MATH|BIO|ASTR|CSC|SCI))\d+", dir_path).group()
        course_name = re.search("^[a-zA-Z]+", course_name).group() + " " + re.search("\d+", course_name).group()
        # print(course_name)
        section = "Section " + re.search("(?<=-)\d+", dir_path).group()
        # print(section)
        folder_dic["season"] = season
        folder_dic["year"] = year
        folder_dic["course_name"] = course_name
        folder_dic["section"] = section
    except:
        print("Error in get_folder_names(). New folder creation unsuccessful.")

    if os.path.isdir(f"./output/{theme}/{folder_dic['season']} {folder_dic['year']}/{folder_dic['course_name']}/{folder_dic['section']}"):
        print("Folder already exists")

    else:
        os.makedirs(f"./output/{theme}/{folder_dic['season']} {folder_dic['year']}/{folder_dic['course_name']}/{folder_dic['section']}")
        print(f"Successfully created ./output/{folder_dic['season']} {folder_dic['year']}/{folder_dic['course_name']}/{folder_dic['section']} folder")
    
    out_path = f"/output/{theme}/{folder_dic['season']} {folder_dic['year']}/{folder_dic['course_name']}/{folder_dic['section']}"
    
    return out_path

# Main CCT Identification Method
def culturalcapitals(data, thematic_code):
    
    # Load the json from cultural_capital_identification()
    appData = json.loads(data)
 
    theme_present = 0
    theme_missing = 0

    # Initialize result table
    result_data_table = {}

    #thematic codes
    themes = {
        2: "Familial",
        # 1: "Attainment",
        # 2: "Aspirational",
        # 3: "Navigational",
        # 4: "Perseverance",
        # 5: "Resistance",
    }

    for i in appData:
        result_data_table["Essay ID"] = [x["Essay ID"] for x in appData]
        result_data_table["Year"] = [x["Year"] for x in appData]
        result_data_table["Semester"] = [x["Semester"] for x in appData]
        result_data_table["Class"] = [x["Class"] for x in appData]
        result_data_table["Type"] = [x["Type"] for x in appData]
        result_data_table["Section"] = [x["Section"] for x in appData]
        result_data_table["Alma ID"] = [x["Alma ID"] for x in appData]

    result_data_table = pd.DataFrame(result_data_table)

    cct_predictions = []
    marked_essays = []
    marked_substrings = []
    
    # THEME MODEL CCT Identification

    # For each element item in the appData
    for item in appData:
        # retrieve and prepare essay to be pushed to the model
        essay = str(item['Essay'])
        # split the essay into sentences
        sentence_list = re.split('\.|\?|\!', essay)
        # get the predictions for the sentence list
        y_pred = models_dict[thematic_code].predict(sentence_list)
        # checking if we found theme
        if(int(max(y_pred)) > 0):
            theme_present = theme_present + 1
            cct_predictions.append('Yes')
            # if theme present, then mark the sentences containing theme.
            for i in range(len(y_pred)):
                if int(y_pred[i]) == 1:
                    marked_str = '<mark>' + str(sentence_list[i]) + '<mark>'
                    marked_substrings.append(marked_str)
                    essay = essay.replace(
                        str(sentence_list[i]), marked_str)
            marked_essays.append(essay)
        else:
            theme_missing = theme_missing + 1
            cct_predictions.append('No')
            marked_essays.append(essay)
    
    # Append predictions and marked essays in result table
    result_data_table[f"{themes[thematic_code]} Present"] = cct_predictions
    result_data_table["Annotated Essays"] = marked_essays
   
    

    return result_data_table
   
# CCT Identification Wrapper
def cultural_capital_identification(input_path, output_path, thematic_code):
    print("Cultural Capital Identification")
    print("input_path", input_path)
    print("output_path", output_path)
    print("thematic code", thematic_code)

    # Step 1: Reading
    csv_read = pd.read_csv(input_path)                   # read the csv
    """Arithmetic
    This method only accepts csvs with the following features:
    * Year
    * Semester
    * Class
    * Type
    * Section
    * Alma ID
    * Essay: Why I am here?
    """
    csv_columns = csv_read.columns.to_list()             # get list of column names
    csv_name = os.path.basename(input_path)[:-4]                 # get the name of the csv
    csv_modify = pd.DataFrame()

    # Step 2: Build csv to be modified
    csv_modify = pd.DataFrame()
    csv_modify["Essay ID"] = generate_essay_id(csv_read)        # generate essay-level id
    csv_modify["Year"] = csv_read["Year"]
    csv_modify["Semester"] = csv_read["Semester"]
    csv_modify["Class"] = csv_read["Class"]
    csv_modify["Type"] = csv_read["Type"]
    csv_modify["Section"] = csv_read["Section"]
    csv_modify["Alma ID"] = csv_read["Alma ID"]
    csv_modify["Essay"] = csv_read.filter(regex=("Essay.*")).squeeze() # match columns that begin with "Essay..."

    
    csv_length = csv_modify.shape[0]
    lens_of_csvs.append(csv_length)

    print("no. of essays", csv_length)
    
    # Step 3: CCT Identification 
    # Decided to keep the app data into json format in case needed in future
    app_data = csv_modify.to_json(orient="records")
   
    # Main meat of the code: calls the culturalcapital() that uses the distillbert model to identify cct
    result = culturalcapitals(app_data, thematic_code)
    # marked essays are the tacit-annonated essays
    marked_essays = result["Annotated Essays"].to_list()
    
    # Step 3. Marking and Exporting to Excel
    # Before we export to excel, we need to mark up the essay so its easier for the reader to find the most relevant sentences of cct-present essays.
    
    #Create a writer object via pandas
    writer = pd.ExcelWriter(f"./{output_path}/" + str(csv_name) + f"_{themes[thematic_code]}_tacited.xlsx", engine='xlsxwriter')

    #Opening the workbook
    result.to_excel(writer, sheet_name="Sheet1")
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    
    # Declare format objects
    bold = workbook.add_format({'bold': True, 'color': 'red'})
    text_wrap = workbook.add_format({'text_wrap': True})

    ## Logic for highlighting marked essays
    for i, string in enumerate(marked_essays):
        # xlsxwriter has a specific syntax on how to write in rich text using write_rich_string: https://xlsxwriter.readthedocs.io/worksheet.html#worksheet-write-rich-string
        # basic rule: ex. worksheet.write_rich_string('A1', 'This is an ', format, 'example', ' string')
        
        # Used <mark> as the delimiter for strings that need to be marked.
        # Split the essay by the delimiter
        segments = re.split(r'<mark>', string)
        # Get list of strings that are marked strings
        marked = re.findall("<mark>"+'(.+?)'+"<mark>", string)


        if len(segments) == 1:
            ## write(Row Index + 1, Column J Index, the string to write)
            ##The i+1 and 9 correspond to the J2 cell, where all of our annonated essays start.
            worksheet.write(i+1, 9, string)
        else:
            # iterate through the segments and add a format before the match
            tmp_array = []
            for segment in segments:
                if segment in marked:
                    tmp_array.append(bold)
                    tmp_array.append(segment)
                elif not segment:
                    pass
                else:
                    tmp_array.append(segment)

            worksheet.write_rich_string(i+1, 9, *tmp_array)
    
    
    human_annotation_familial_headers = ["Human Theme Presence (Yes or No)", "Human Annotated Essay", "Specific Theme(s)"]
    # Write the headers to columns K, L, and M
    worksheet.write('K1', human_annotation_familial_headers[0])
    worksheet.write('L1', human_annotation_familial_headers[1])
    worksheet.write('M1', human_annotation_familial_headers[2])

    # Autofit and text warp 
    worksheet.set_column('J:J', None, text_wrap)
    worksheet.autofit()
    workbook.close()
    
    out_file_name = f"{output_path}/" + str(csv_name) + f"_{themes[thematic_code]}_tacited.xlsx"
    return result, out_file_name

# Using columns in a csv, generate an essay-level id for each row/essay.
def generate_essay_id(csv):
    essay_id_list = []

    for i,j in csv.iterrows():
        ### sample id F19.PHYS223.01.02
        year = str(j["Year"])[-2:]
        class_name = "".join(j["Class"].split())
        semester = j["Semester"][0]
        section = str(j["Section"]).zfill(2)
        row_no = str(i).zfill(3)
        alma_id = str(j["Alma ID"]).zfill(3)
        
        row_id = f'{semester}{year}.{class_name}.{section}.{row_no}.{alma_id}'
        essay_id_list.append(row_id)

    return essay_id_list

# Start the process
if __name__ == "__main__":
    import pandas as pd
    import gensim
    import spacy
    import json
    import re
    import glob
    import sys
    import os
    import time
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    from sklearn.feature_extraction.text import CountVectorizer

    # ktrain to load model
    import ktrain
    # Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # Load the attainment model at the startup.
    # attainment_model = ktrain.load_predictor('../tacit-backend/distillbert_model/')
    familial_model = ktrain.load_predictor("/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/saved_models/familial_distilbert_base_uncased_model_07242024") 
    # navigational_model = ktrain.load_predictor("./model/nav_distilbert_base_uncased_model/") 
    # perseverance_model = ktrain.load_predictor("../perseverance-resistance/model/per_distilbert_base_uncased_model/") 
    # resistance_model = ktrain.load_predictor("../perseverance-resistance/model/res_distilbert_base_uncased_model/") 
    
    models_dict = {
        2: familial_model,
        # 3: navigational_model,
        # 4: perseverance_model,
        # 5: resistance_model
    }
    # distillbert is a light version of bert model...

    # Initialize Count Vectorizer at the start.
    vectorizer = CountVectorizer(analyzer='word',
                                # min_df=1,                      # minimum reqd occurences of a word
                                # stop_words='english',           # remove stop words
                                lowercase=True)                 # convert all words to lowercase
    # token_pattern='[a-zA-Z0-9]{2,}')# num chars > 3
    # max_features=50000,           # max number of uniq words

    ## load the input parameters
    input_dir_path  = sys.argv[1]
    overwrite = sys.argv[2]

    print("input_dir_path =", sys.argv[1])
    print("overwrite", overwrite)

    input_dir = sorted(glob.glob(f"{input_dir_path}/**/*.csv", recursive=True))

    lens_of_csvs = []

    #thematic codes
    themes = {
        # 1: "Aspirational",
        2: "Familial",
        # 3: "Navigational",
        # 4: "Perseverance",
        # 5: "Resistance",
    }


    for theme_key, theme_val in themes.items():
        print(f"{theme_val} Analysis")
        print(f"\n")
        start_time = time.time()
        for i in input_dir:
            print(f"Theme: {theme_val}")
            # Get the original csv name without extension
            csv_name = os.path.basename(i)[:-4]
            
            # # Extract part of the filename after the first underscore
            # match = re.search(r'_(.*)', original_csv_name)
            # if match:
            #     csv_name = match.group(1)
            # else:
            #     csv_name = original_csv_name  # Fallback in case the regex doesn't match
            
            output_path = generate_folder(i, theme_val)
          
            out_file_name = f"{output_path}/" + str(csv_name) + f"_{themes[2]}_tacited.xlsx"
            print("/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/src/tacit/Familial" + out_file_name)
            # print("HEY HERE", os.path.isfile(out_file_name))
            # print("HEY HERE", out_file_name)
            if overwrite == "False" and os.path.isfile("/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/src/tacit/Familial" + out_file_name):
                print(f"File {csv_name} already exists. No overwriting was performed. Terminating process for this file.")
            else:
                _, out_file_name = cultural_capital_identification(i, output_path, theme_key)
                print(out_file_name + " CCT identified and spreadsheet written successfully.")
    
        print(f"---{time.time() - start_time} seconds ---")
        print("\n")
       
        print("No of essays labeled:", sum(lens_of_csvs))