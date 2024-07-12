import os
import pandas as pd
import xlsxwriter

new_columns = [
    'Attainment', 'Aspirational', 'First Generation', 'Navigational',
    'Resistance', 'Perseverance', 'Familial', 'Filial Piety',
    'Community Consciousness', 'Social', 'Spiritual'
]

def convert_csv_to_xlsx(csv_file, xlsx_file, new_columns):
    df = pd.read_csv(csv_file)

    # Create a new DataFrame with the new columns and fill them with empty values
    new_data = {col_name: [''] * len(df) for col_name in new_columns}
    new_df = pd.DataFrame(new_data)

    # Merge the new DataFrame with the existing data
    df = pd.concat([df, new_df], axis=1)

    # Write to XLSX file with text wrapping
    writer = pd.ExcelWriter(xlsx_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    worksheet = writer.sheets['Sheet1']

    for col_name in new_columns:
        col_index = df.columns.get_loc(col_name)
        worksheet.set_column(col_index, col_index, None, None, {'text_wrap': True})

    writer.save()

def process_csv_files(root_directory, new_columns):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                xlsx_file = os.path.splitext(csv_file)[0] + '.xlsx'
                convert_csv_to_xlsx(csv_file, xlsx_file, new_columns)

root_directory = '/path/to/your/csv/files'  # Replace with the path to your CSV files
process_csv_files(root_directory, new_columns)
