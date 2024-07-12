'''
INSTRUCTIONS:
1. Go to folder where script is in the terminal.
2. In terminal, run the following:
    python3 anonymize.py <input_path_file> <key_file> <overwrite?>
    ex. input/Fall2020_PHYS0102-02_Essay1.csv alma_id_mapping.txt False

PURPOSE:
As of 02.02.2023, the ALMA IDs assigned to students are not mapped to their student IDs. 
For example, in one CSV, student 912345678 is mapped to ALMA ID 1, while in another CSV, 
another student ID is mapped to ALMA ID 1. This script aims to solve this problem through a program 
that takes in two inputs: (1) the CSV that needs to be modified, and (2) a CSV or TXT file that holds 
the mappings of the ALMA IDs and student IDS.
'''

def jsonKeys2int(x):
    """
    Helper function to reconstruct json keys into ints. Used to reconstruct key file to a dictionary
    :param: x = jsonified dictionary from a text file
    """
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def generate_folder(dir_path):
    """
    Creates hierarchy of directories for the output.
    Follows the structure: [Root Output] > [Semester Year] > [Courese Name and ID] > [Section] > [*.csv]
    """
    folder_dic = {"season": "", "year": "", "course_name": "", "section": ""}
   
    try:
        season = re.search("Fall|Spring|FALL|SPRING|SUMMER|Summer", dir_path).group()
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

    if os.path.isdir(f"./output/{folder_dic['season']} {folder_dic['year']}/{folder_dic['course_name']}/{folder_dic['section']}"):
        print("Folder already exists")
    else:
        os.makedirs(f"./output/{folder_dic['season']} {folder_dic['year']}/{folder_dic['course_name']}/{folder_dic['section']}")
        print(f"Successfully created ./output/{folder_dic['season']} {folder_dic['year']}/{folder_dic['course_name']}/{folder_dic['section']} folder")
    
    out_path = f"./output/{folder_dic['season']} {folder_dic['year']}/{folder_dic['course_name']}/{folder_dic['section']}"
    
    return out_path

def anonymize_csv(input_path, output_path, key_path, first=False):
    """
    Takes in path to a csv file, an output path, and the path to the key file.
    """
    print("Anonymization CSV function")
    print("input_path", input_path)
    print("output_path", output_path)
    print("key_path", key_path)

    # Step 1: Reading
    csv_to_search = pd.read_csv(input_path, encoding='utf-8')                 # read the csv
    sfsu_id_list = csv_to_search["SF State ID"].to_list()   # get list of SF State IDs
    csv_modify = csv_to_search.copy()                       # make a copy of the input csv
    csv_name = os.path.basename(input_path)[:-4]                 # get the name of the csv
    id_mapping = {}

    out_file_name = f"{output_path}/" + str(csv_name) + "_anonymized.csv"
    
    with open(key_path) as key_file:                        # read the alma key file
        data = key_file.read()

    if data:
        id_mapping = json.loads(data)
        id_mapping = jsonKeys2int(id_mapping)               # reconstruct key file from string to dictionary
    
    original_id_mapping = id_mapping.copy()
    
    # save a copy of last alma id mapping
    with open(f"./id_mapping_history/id_mapping_{dt_string}.txt", 'w') as key_file:     # save a copy of last alma id mapping
        key_file.write(json.dumps(original_id_mapping))
        print(f"Copy of id_mapping on {dt_string} saved")
    
    # Step 2: ID assignment
    ## The idea: first ID read is 1, second ID is alma ID 2 
    if first:                                               # default false. first="true" if alma key file is empty.
        last_alma_id = 0                                    # the first sfsu id read is mapped to alma id 1
    else:
        last_alma_id = max(id_mapping.values())             # the max id of the alma key file is the previous id file 
        
    for i in sfsu_id_list:                                  # for each id
        if i not in id_mapping.keys():                      # if current id is not in the alma key file
            last_alma_id += 1                                  # assign an alma id that is previous alma id + 1
            id_mapping[i] = last_alma_id                       # save the new alma id 
    
    for i in csv_modify.index:                              # for each row/index in the copy/to-be-modified csv
        csv_modify.loc[i, ["Alma ID"]] = id_mapping[csv_modify.loc[i, ["SF State ID"]].item()] # assign the "Alma ID" of that row to the Alma ID mapped to that row's "SF State ID"
    
    # Step 3: Return modified csv and updated id_mapping
    with open('alma_id_mapping.txt', 'w') as key_file:     # update the alma key file 
        key_file.write(json.dumps(id_mapping))
    
    # print("id_mapping", id_mapping)
    
    csv_modify.drop(columns="SF State ID", inplace=True)   # drop the SF State ID in the modified csv
    csv_modify.to_csv(f"{output_path}/" + str(csv_name) + "_anonymized.csv", index=False) # output modified csv to the output folder

    return csv_modify, out_file_name, id_mapping



# start the process
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import pickle
    import random
    import json
    import datetime
    import re
    import glob
    import sys
    import os
    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    ## load the input parameters
    input_dir_path  = sys.argv[1]
    key_path = sys.argv[2]
    overwrite = sys.argv[3] if len(sys.argv) >= 4 else "False"

    print("input_dir_path =", sys.argv[1])
    print("key_path = ", sys.argv[2])
    # print("overwrite", sys.argv[3])

    # create a list of inputs
    input_dir = sorted(glob.glob(f"{input_dir_path}/**/*.csv", recursive=True))

    for i in input_dir:
        csv_name = os.path.basename(i)[:-4]                 # get the name of the csv
        output_path = generate_folder(i)
        out_file_name = f"{output_path}/" + str(csv_name) + "_anonymized.csv"
        print("IM HERE")
        if overwrite == "False" and os.path.isfile(out_file_name) == True:
            print(f"File {csv_name} already exists. No overwriting was performed. Terminating process for this file.")
        else:
            __, out_file_name, __ = anonymize_csv(i, output_path, key_path)
            print(out_file_name + " written successfully.")