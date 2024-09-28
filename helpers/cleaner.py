import pandas as pd

# import excel file
file_path = '../data/data.xlsx' 
df = pd.read_excel(file_path)

# check for missing values
print(df.isnull().sum()) 

# drop rows with missing perfume value
df_cleaned = df.dropna(subset=['perfume'])

# normalize note column
df_cleaned.loc[:, 'notes'] = df_cleaned['notes'].str.lower().str.strip()  

# remove duplicate
df_cleaned = df_cleaned.drop_duplicates(subset=['perfume', 'brand', 'notes'])

# split notes into a list
df_cleaned.loc[:, 'notes'] = df_cleaned['notes'].apply(lambda x: x.split(', '))

# save new cleaned list
df_cleaned.to_excel('../data/cleaned_perfume_data.xlsx', index=False)

print("Cleaned data has been saved to 'cleaned_perfume_data.txt'")
