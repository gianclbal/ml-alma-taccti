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

How to run
python3 -m memory_profiler tacit-app_aspirational.py /Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/data/to_be_tacited/sample_data False
python3 -m memory_profiler tacit-app_aspirational.py /Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/data/to_be_tacited/fall_2020_and_spring_2020 False
python3 -m memory_profiler tacit-app_aspirational.py /Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/data/to_be_tacited/batch_2 False


DATE:
March 17, 2023
'''


@profile 
# Sentence to words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# Lemmatization
# 'NOUN', 'ADJ', 'VERB', 'ADV'
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    for sent in texts:
        doc = nlp(" ".join(sent))
        yield " ".join([token.lemma_ if token.lemma_ not in [
                         '-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])

def preprocess_text(lst):
    paragraphs = [str(x or "") for x in lst]
    paragraphs = re.sub(r'\S*@\S*\s?', '', '\n'.join(paragraphs)).splitlines()

    text_words = sent_to_words(paragraphs)
    text_lemmatized = lemmatization(text_words)

    for text_lem in text_lemmatized:
        # Process text_lem using vectorizer and other steps
        # Yield or accumulate results as needed
        pass

    # Explicitly delete large objects
    del paragraphs, text_words, text_lemmatized
    gc.collect()

@profile 
# Get the string between two characters/words
def getSubstring(ch1,ch2,s):
    m = re.findall(ch1+'(.+?)'+ch2, s)
    return 
@profile 
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

@profile
def culturalcapitals(data, thematic_code):
    appData = json.loads(data)
    themes = {1: "Aspirational"}

    result_data_table = {
        "Essay ID": (x["Essay ID"] for x in appData),
        "Year": (x["Year"] for x in appData),
        "Semester": (x["Semester"] for x in appData),
        "Class": (x["Class"] for x in appData),
        "Type": (x["Type"] for x in appData),
        "Section": (x["Section"] for x in appData),
        "Alma ID": (x["Alma ID"] for x in appData),
    }

    result_data_table = pd.DataFrame(result_data_table)

    cct_predictions = []
    marked_essays = []
    marked_substrings = []

    theme_present = 0
    theme_missing = 0

    for item in appData:
        essay = str(item['Essay'])
        sentence_list = re.split('\.|\?|\!', essay)
        y_pred = models_dict[thematic_code].predict(sentence_list)

        if int(max(y_pred)) > 0:
            theme_present += 1
            cct_predictions.append('Yes')
            for i in range(len(y_pred)):
                if int(y_pred[i]) == 1:
                    marked_str = '<mark>' + str(sentence_list[i]) + '<mark>'
                    marked_substrings.append(marked_str)
                    essay = essay.replace(str(sentence_list[i]), marked_str)
            marked_essays.append(essay)
        else:
            theme_missing += 1
            cct_predictions.append('No')
            marked_essays.append(essay)

    result_data_table[f"{themes[thematic_code]} Present"] = cct_predictions
    result_data_table["Annotated Essays"] = marked_essays

    del appData
    gc.collect()

    return result_data_table


@profile
def cultural_capital_identification(input_path, output_path, thematic_code):
    print("Cultural Capital Identification")
    print("input_path", input_path)
    print("output_path", output_path)
    print("thematic code", thematic_code)

    csv_read = pd.read_csv(input_path)
    csv_name = os.path.basename(input_path)[:-4]

    csv_modify = pd.DataFrame({
        "Essay ID": generate_essay_id(csv_read),
        "Year": csv_read["Year"],
        "Semester": csv_read["Semester"],
        "Class": csv_read["Class"],
        "Type": csv_read["Type"],
        "Section": csv_read["Section"],
        "Alma ID": csv_read["Alma ID"],
        "Essay": csv_read.filter(regex=("Essay.*")).squeeze()
    })

    csv_length = csv_modify.shape[0]
    print("no. of essays", csv_length)

    app_data = csv_modify.to_json(orient="records")
    result = culturalcapitals(app_data, thematic_code)
    marked_essays = result["Annotated Essays"].to_list()

    with pd.ExcelWriter(f"./{output_path}/{csv_name}_{themes[thematic_code]}_tacited.xlsx", engine='xlsxwriter') as writer:
        result.to_excel(writer, sheet_name="Sheet1")
        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]
        bold = workbook.add_format({'bold': True, 'color': 'red'})
        text_wrap = workbook.add_format({'text_wrap': True})

        for i, string in enumerate(marked_essays):
            segments = re.split(r'<mark>', string)
            marked = re.findall("<mark>"+'(.+?)'+"<mark>", string)

            if len(segments) == 1:
                worksheet.write(i + 1, 9, string)
            else:
                tmp_array = []
                for segment in segments:
                    if segment in marked:
                        tmp_array.append(bold)
                        tmp_array.append(segment)
                    elif not segment:
                        pass
                    else:
                        tmp_array.append(segment)
                worksheet.write_rich_string(i + 1, 9, *tmp_array)

        worksheet.write('K1', "Human Theme Presence (Yes or No)")
        worksheet.write('L1', "Human Annotated Essay")
        worksheet.write('M1', "Specific Theme(s)")

        worksheet.set_column('J:J', None, text_wrap)
        worksheet.autofit()

    del result, marked_essays, csv_modify, csv_read
    gc.collect()

    return f"{output_path}/{csv_name}_{themes[thematic_code]}_tacited.xlsx"

@profile
def generate_essay_id(csv):
    return [
        f'{j["Semester"][0]}{str(j["Year"])[-2:]}.{j["Class"].replace(" ", "")}.{str(j["Section"]).zfill(2)}.{str(i).zfill(3)}.{str(j["Alma ID"]).zfill(3)}'
        for i, j in csv.iterrows()
    ]


# Start the process
if __name__ == "__main__":
    from memory_profiler import profile
    import pandas as pd
    import gensim
    import spacy
    import json
    import re
    import glob
    import sys
    import os
    import gc
    import time
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    from sklearn.feature_extraction.text import CountVectorizer

    # ktrain to load model
    import ktrain
    # Initialize spacy ‘en’ model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # Load the attainment model at the startup.
    # attainment_model = ktrain.load_predictor('../tacit-backend/distillbert_model/')
    # familial_model = ktrain.load_predictor("/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/saved_models/familial_distilbert_base_uncased_model_07242024")
    # social_model = ktrain.load_predictor("/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/saved_models/social_bert_base_cased_model_08012024")  
    # navigational_model = ktrain.load_predictor("./model/nav_distilbert_base_uncased_model/") 
    # perseverance_model = ktrain.load_predictor("../perseverance-resistance/model/per_distilbert_base_uncased_model/") 
    # resistance_model = ktrain.load_predictor("../perseverance-resistance/model/res_distilbert_base_uncased_model/") 
    # resistance_model = ktrain.load_predictor("/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/saved_models/resistance_bert_base_cased_model_08192024")
    aspirational_model = ktrain.load_predictor("/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/saved_models/aspirational_bert_base_cased_model_08012024_v1")

    
    models_dict = {
        1: aspirational_model,
        # 2: familial_model,
        # 3: social_model,
        # 4: resistance_model,
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
        1: "Aspirational",
        # 2: "Familial",
        # 3: "Social",
        # 4: "Resistance",
        # 5: "Resistance",
    }

    # Manually set the desired essay number here
    desired_essay_number = "Essay1"  # Change this to "Essay2", "Essay3", etc. as needed

    for theme_key, theme_val in themes.items():
        print(f"{theme_val} Analysis")
        print(f"\n")
        start_time = time.time()

         # Filter input_dir for files containing the desired essay number
        filtered_input_dir = [
            i for i in input_dir 
            if desired_essay_number in os.path.basename(i) and i.endswith('.csv')
        ]
        
        for i in filtered_input_dir:
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
          
            out_file_name = f"{output_path}/" + str(csv_name) + f"_{themes[1]}_tacited.xlsx"
            print("/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/src/tacit/Aspirational" + out_file_name)
            # print("HEY HERE", os.path.isfile(out_file_name))
            # print("HEY HERE", out_file_name)
            if overwrite == "False" and os.path.isfile("/Users/gbaldonado/Developer/ml-alma-taccti/ml-alma-taccti/src/tacit/Aspirational" + out_file_name):
                print(f"File {csv_name} already exists. No overwriting was performed. Terminating process for this file.")
            else:
                out_file_name = cultural_capital_identification(i, output_path, theme_key)
                print(out_file_name + " CCT identified and spreadsheet written successfully.")

             # Explicitly delete large objects
            del csv_name, output_path, out_file_name
            gc.collect()  # Collect garbage
    
        print(f"---{time.time() - start_time} seconds ---")
        print("\n")
       
        print("No of essays labeled:", sum(lens_of_csvs))