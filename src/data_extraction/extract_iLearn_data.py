#!/usr/bin/python3
#####################################################################################
# Name: extract_iLearn_data.py                                                      #
# Description: Python script to extract data from iLearn zip files.                 #
# Author: Karuna Nayak & Shailesh Krishna                                           #
#####################################################################################

'''
INSTRUCTIONS:
1. This script handles both the MacOS and Windows downloaded zip files.
2. Download Data
   * Before running this script, download the zip file from iLearn portal.
3. Create folder/directory on your machine.
   * python3 extract_iLearn_data.py <input_directory_path> <output_directory_path> <input_file_ids.xlsx> <input_file_name>
   * Example: python3 extract_iLearn_data.py /usr/input /usr/output PHYS022204-F18RGrades.xlsx PHYS022204-F18R-ReflectionEssay#1-184302.zip
* python3 extract_iLearn_data.py input/FALL_2020/PHYS010202-F20R output/ PHYS010202-F20R-3073_Grades.xlsx PHYS010202-F20R-3073-Reflection_Essay_#1-208332.zip 
PURPOSE:
1. This script extracts the essays and its metadata from the zip file downloaded from iLearn.
'''
########################################################################################
# Function Name: create_name_dict                                                      #
# Description: Create a dictionary containing SFSU ids.                                #
########################################################################################
def create_name_dict(id_file_name):
    df = pd.read_csv(id_file_name)
    #df = pd.read_csv(id_file_name)
    name_dict = {}
    for index, row in df.iterrows():
        name = re.sub('\'', '', row['First name']) + " " + re.sub('\'', '',row["Surname"])
        if name in name_dict.keys():
            print("Same name student exists! " + name)
        else:
            name_dict[name] = row["Username"]

    return name_dict

########################################################################################
# Function Name: create_essay_csv                                                      #
# Description: Create the csv file containing the essays.                              #
########################################################################################
def create_essay_csv(in_folder, out_path, name_dict, year, sem, sub, course_num, class_type, section):
    with open( out_path, 'w', newline='', errors='ignore', ) as file:
        sfid_list = [] # sfsu id list
        essay_writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL) # writer method.
        essay_writer.writerow(['\ufeffYear', 'Semester', 'Class', 'Type', 'Section', 'Alma ID', 'SF State ID', essay_prompt]) #  header row
        cnt = 1 # counter
        for directory in os.listdir(in_folder): # read the directory
            #print("file name: ",cnt , " ",  directory)
            if directory == ".DS_Store": # ignore directory starting with this name.
                continue

            try:
                # get the student name from the name of the folder.
                std_name = re.search('(.*)_\d+_.*', directory ).group(1)
            except :
                # unable to find the name.
                print("name coudnt find in " + directory)

            try:
                # lookup the student id in the dictionary created previously.
                sfid = name_dict[std_name]
            except:
                print("Student name " + std_name + " does not exist on ilearn grading sheet")
                # if the student name does not exist in the dictionary, then assign a dummy student id.
                sfid = 111111111

            if sfid in sfid_list:
                print("Duplicate") # if id exists then continue.
                continue
            else:
                student_dir_path = os.path.join(in_folder, directory)
                for essay_file in os.listdir(student_dir_path):
                    file_path = os.path.join(student_dir_path,essay_file)
                    if essay_file.endswith('.html'):
                        html_file = open(file_path , 'r', encoding='UTF-8')
                        text = html.fromstring(html_file.read()).text_content()
                        html_file.close()
 
                    if essay_file.endswith('.docx'):
                        doc = Document(file_path)
                        para_text = []
                        for para in doc.paragraphs:
                            para_text.append(para.text)
                        text = '\n'.join(para_text)

                    if essay_file.endswith('.pdf'):
                        # print(file_path)
                        pdfObj = open(file_path, 'rb')
                        pdfReader = PyPDF2.PdfReader(pdfObj)
                        page_obj = pdfReader.pages[0]
                        text = page_obj.extract_text()
                        pdfObj.close()

                    #clean the extarcted text
                    if(text != None or len(text) >2):
                        sfid_list.append(sfid)
                        text = text.lstrip()
                        text = text.rstrip('\r \n')
                        text= re.sub('\n',' ', text)
   
                    essay_writer.writerow([year, sem, sub+" "+course_num, class_type, section, cnt , sfid, text])

                    cnt += 1
                    text = ''

# start the process
if __name__ == "__main__":
    
    import sys
    import os
    from zipfile import ZipFile # requires installation
    import re
    import shutil # filesystem
    # pip install python-docx
    from docx import Document # read .doc # requires installation 
    from lxml import html
    import csv
    import pandas as pd # requires installation
    import json
    import PyPDF2 # requires installation

    ## load the input parameters
    input_path  = sys.argv[1]
    out_dir = sys.argv[2]
    id_file_name = sys.argv[3]
    in_file_name = sys.argv[4]

    print("input_path  =", sys.argv[1])
    print("out_dir = ", sys.argv[2])
    print("in_file_name = ", sys.argv[3])
    print("id_file_name = ",sys.argv[4])

    #extract the class meta info from the file name
    sub = re.search('(\w+?)\d', in_file_name).group(1)
    section = re.search('(\d{2}?)\-', in_file_name).group(1)
    course_num = re.search('{}(\d+){}'.format(sub, section), in_file_name).group(1)
    sem_char = re.search('\-(\w?)\d', in_file_name).group(1)
    sem = ("Summer" if sem_char == 'U' else
       "Fall" if sem_char == 'F' else
       "Spring")

    year = '20'+ re.search('\-{}(\d\d)'.format(sem_char), in_file_name).group(1)
    #essay_num = re.search('.*Essay #(\d+)-.*', in_file_name).group(1)
    essay_num = re.search('#(\d+)', in_file_name).group(1)
    

    if int(essay_num) == 1:
        essay_prompt = "Essay: Why I am here?"
    else:
        essay_prompt = "Essay: "+essay_num

    out_file_name = sem+year+"_"+sub+course_num+"-"+str(section)+"_"+"Essay"+essay_num+'.csv'
    out_path = os.path.join(out_dir, out_file_name)

    # load the json file with class codes
    with open('classCodes.json') as c_file:
        class_codes = json.load(c_file)
    try:
        class_type =   class_codes[sub+" "+course_num][str(section)]
    except:
        if sub=="SCI" :
            class_type = "SI"
        else:
            class_type = "--"
    in_path = os.path.join(input_path,in_file_name)
    name_dict = create_name_dict(os.path.join(input_path, id_file_name))


    if in_file_name.endswith(".zip"):
        ZipFile(in_path).extractall('temp') # extract the zip file into a temp folder
        temp_path = os.path.join(os.getcwd(), 'temp')
        create_essay_csv(temp_path, out_path, name_dict, year, sem, sub, course_num, class_type, section)
        shutil.rmtree("temp")

    else:
        create_essay_csv(in_path, out_path, name_dict, year, sem, sub, course_num, class_type, section)


    print(out_file_name + " created successfully")


