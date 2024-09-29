import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load categorized perfumes data
df = pd.read_excel('./data/categorized_perfumes.xlsx')

# load unique notes sheet
with open('./data/unique_notes_cleaned.csv', 'r') as f:
    unique_notes = [line.strip().strip('"') for line in f]

# load pre trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# encode unique notes
unique_notes_embeddings = model.encode(unique_notes)

# define note compatibility matrix (not arbitrary, based on expert (me) knowledge)
compatibility_matrix = {
    'Floral': {'Floral': 0.9, 'Fruity': 0.7, 'Woody': 0.7, 'Spicy': 0.6, 'Citrus': 0.8, 'Herbal': 0.65, 'Sweet': 0.65, 'Earthy': 0.4, 'Aquatic': 0.9, 'Gourmand': 0.25},
    'Fruity': {'Floral': 0.7, 'Fruity': 0.7, 'Woody': 0.1, 'Spicy': 0.4, 'Citrus': 0.3, 'Herbal': 0.5, 'Sweet': 0.65, 'Earthy': 0.3, 'Aquatic': 0.8, 'Gourmand': 0.75},
    'Woody': {'Floral': 0.8, 'Fruity': 0.1, 'Woody': 0.9, 'Spicy': 0.85, 'Citrus': 0.8, 'Herbal': 0.5, 'Sweet': 0.6, 'Earthy': 0.9, 'Aquatic': 0.75, 'Gourmand': 0.4},
    'Spicy': {'Floral': 0.3, 'Fruity': 0.1, 'Woody': 0.8, 'Spicy': 0.9, 'Citrus': 0.65, 'Herbal': 0.6, 'Sweet': 0.65, 'Earthy': 0.6, 'Aquatic': 0.3, 'Gourmand': 0.45},
    'Citrus': {'Floral': 0.8, 'Fruity': 0.3, 'Woody': 0.65, 'Spicy': 0.6, 'Citrus': 0.9, 'Herbal': 0.5, 'Sweet': 0.6, 'Earthy': 0.3, 'Aquatic': 0.8, 'Gourmand': 0.4},
    'Herbal': {'Floral': 0.8, 'Fruity': 0.35, 'Woody': 0.8, 'Spicy': 0.4, 'Citrus': 0.8, 'Herbal': 0.9, 'Sweet': 0.5, 'Earthy': 0.7, 'Aquatic': 0.75, 'Gourmand': 0.1},
    'Sweet': {'Floral': 0.55, 'Fruity': 0.65, 'Woody': 0.6, 'Spicy': 0.3, 'Citrus': 0.65, 'Herbal': 0.2, 'Sweet': 0.9, 'Earthy': 0.1, 'Aquatic': 0.1, 'Gourmand': 0.85},
    'Earthy': {'Floral': 0.65, 'Fruity': 0.25, 'Woody': 0.9, 'Spicy': 0.75, 'Citrus': 0.4, 'Herbal': 0.7, 'Sweet': 0.45, 'Earthy': 0.9, 'Aquatic': 0.5, 'Gourmand': 0.6},
    'Aquatic': {'Floral': 0.8, 'Fruity': 0.8, 'Woody': 0.75, 'Spicy': 0.3, 'Citrus': 0.8, 'Herbal': 0.75, 'Sweet': 0.1, 'Earthy': 0.5, 'Aquatic': 0.9, 'Gourmand': 0.1},
    'Gourmand': {'Floral': 0.25, 'Fruity': 0.75, 'Woody': 0.4, 'Spicy': 0.45, 'Citrus': 0.45, 'Herbal': 0.1, 'Sweet': 0.85, 'Earthy': 0.6, 'Aquatic': 0.1, 'Gourmand': 0.9}
}

# find perfume by brand and name (temporary solution)
def find_perfume(brand, name):
    perfume = df[(df['brand'] == brand) & (df['perfume'] == name)]
    if perfume.empty:
        return None
    return perfume.iloc[0]

# get notes sets from both perfume and calculate similarity using cosine similarity
def calculate_note_similarity(notes1, notes2):
    embeddings1 = model.encode(notes1)
    embeddings2 = model.encode(notes2)
    return cosine_similarity(embeddings1.mean(axis=0).reshape(1, -1), 
                             embeddings2.mean(axis=0).reshape(1, -1))[0][0]

# calculate compatibility between two categories using compatibility matrix
def calculate_category_compatibility(categories1, categories2):
    compatibility = 0
    for cat1, score1 in categories1:
        for cat2, score2 in categories2:
            compatibility += compatibility_matrix[cat1][cat2] * score1 * score2
    return compatibility / (len(categories1) * len(categories2))

# calculate overall compatibility between two perfumes combining cosine similarity of notes and category compatibility with matrix
def calculate_compatibility(perfume1, perfume2):
    notes1 = perfume1['notes'].split(',')
    notes2 = perfume2['notes'].split(',')
    
    note_similarity = calculate_note_similarity(notes1, notes2)
    
    categories1 = eval(perfume1['Top Categories'])
    categories2 = eval(perfume2['Top Categories'])
    category_compatibility = calculate_category_compatibility(categories1, categories2)
    
    # combine note similarity and category compatibility, using 60/40 weight ratio
    overall_compatibility = (0.6 * note_similarity + 0.4 * category_compatibility) * 100
    return round(overall_compatibility, 2)

# get perfume compatibility (percentage) between two perfumes
def get_perfume_compatibility(brand1, name1, brand2, name2):
    perfume1 = find_perfume(brand1, name1)
    perfume2 = find_perfume(brand2, name2)
    
    if perfume1 is None or perfume2 is None:
        return "One or both perfumes not found in the database."
    
    compatibility = calculate_compatibility(perfume1, perfume2)
    return f"{name1} by {brand1} and {name2} by {brand2} are {compatibility}% compatible."

# test
print(get_perfume_compatibility("Carolina Herrera", "Good Girl", "Avon", "Incandessence"))