import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_excel('./data/categorized_perfumes.xlsx')

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define compatibility matrix
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

def find_perfume(brand, name):
    perfume = df[(df['brand'] == brand) & (df['perfume'] == name)]
    if perfume.empty:
        return None
    return perfume.iloc[0]

def calculate_note_similarity(notes1, notes2):
    embeddings1 = model.encode(notes1)
    embeddings2 = model.encode(notes2)
    return cosine_similarity(embeddings1.mean(axis=0).reshape(1, -1), 
                             embeddings2.mean(axis=0).reshape(1, -1))[0][0]

def create_category_vector(top_categories, all_categories):
    vector = np.zeros(len(all_categories))
    for category, score in top_categories:
        index = all_categories.index(category)
        vector[index] = score
    return vector

def calculate_category_vector_compatibility(perfume1, perfume2):
    all_categories = list(compatibility_matrix.keys())
    
    top_categories1 = eval(perfume1['Top Categories'])
    top_categories2 = eval(perfume2['Top Categories'])
    
    vector1 = create_category_vector(top_categories1, all_categories)
    vector2 = create_category_vector(top_categories2, all_categories)
    
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    
    return cosine_similarity

def calculate_enhanced_compatibility(perfume1, perfume2):
    notes1 = perfume1['notes'].split(',')
    notes2 = perfume2['notes'].split(',')
    
    note_similarity = calculate_note_similarity(notes1, notes2)
    category_compatibility = calculate_category_vector_compatibility(perfume1, perfume2)
    
    # Combine scores (you can adjust these weights)
    overall_compatibility = (0.5 * note_similarity + 0.5 * category_compatibility) * 100
    
    return round(overall_compatibility, 2)

def get_perfume_compatibility(brand1, name1, brand2, name2):
    perfume1 = find_perfume(brand1, name1)
    perfume2 = find_perfume(brand2, name2)
    
    if perfume1 is None or perfume2 is None:
        return "One or both perfumes not found in the database."
    
    compatibility = calculate_enhanced_compatibility(perfume1, perfume2)
    return f"{name1} by {brand1} and {name2} by {brand2} are {compatibility}% compatible."

# Test the function
print(get_perfume_compatibility("Carolina Herrera", "Good Girl", "Avon", "Incandessence"))