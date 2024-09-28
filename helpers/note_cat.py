import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load file
df = pd.read_csv('../unique_notes_cleaned.csv', header=None, names=['note'])
notes = df['note'].tolist()

# create example categories
categories = {
    'Floral': [
        'rose', 'jasmine', 'lily', 'lavender', 'gardenia', 'peony', 'violet', 
        'carnation', 'orchid', 'magnolia', 'tuberose', 'geranium', 'freesia', 
        'lilac', 'cherry blossom', 'orange blossom', 'lotus', 'ylang-ylang', 'iris', 'cotton flower', 'daisy', 'tulip', 'palmarosa', 'edelweiss'
    ],
    'Fruity': [
        'apple', 'strawberry', 'peach', 'pear', 'raspberry', 'blackberry', 
        'mango', 'pineapple', 'apricot', 'plum', 'cherry', 'melon', 'fig', 
        'pomegranate', 'passion fruit', 'guava', 'coconut', 'kiwi'
    ],
    'Woody': [
        'cedar', 'sandalwood', 'pine', 'oak', 'vetiver', 'patchouli', 
        'rosewood', 'agarwood (oud)', 'cypress', 'birch', 'teak', 'ebony', 
        'mahogany', 'juniper', 'fir', 'bamboo', 'driftwood', 'cashmere wood'
    ],
    'Spicy': [
        'cinnamon', 'pepper', 'ginger', 'cardamom', 'clove', 'nutmeg', 
        'saffron', 'cumin', 'coriander', 'anise', 'star anise', 'fennel', 
        'caraway', 'pimento', 'allspice', 'paprika', 'chili'
    ],
    'Citrus': [
        'lemon', 'orange', 'grapefruit', 'lime', 'bergamot', 'tangerine', 
        'mandarin', 'yuzu', 'pomelo', 'citron', 'kumquat', 'clementine', 
        'lemongrass', 'kaffir lime', 'neroli'
    ],
    'Herbal': [
        'mint', 'basil', 'thyme', 'rosemary', 'sage', 'chamomile', 'dill', 
        'oregano', 'tarragon', 'marjoram', 'parsley', 'bay leaf', 'cilantro', 
        'chervil', 'fennel', 'lemongrass', 'verbena'
    ],
    'Sweet': [
        'vanilla', 'caramel', 'honey', 'chocolate', 'sugar', 'toffee', 
        'marshmallow', 'cotton candy', 'maple syrup', 'butterscotch', 
        'licorice', 'praline', 'nougat', 'marzipan', 'meringue'
    ],
    'Earthy': [
        'patchouli', 'moss', 'musk', 'leather', 'soil', 'petrichor', 'truffle', 
        'beet', 'vetiver', 'hay', 'tobacco', 'oakmoss', 'loam', 'humus', 'bark'
    ],
    'Aquatic': [
        'sea salt', 'marine', 'ocean', 'seaweed', 'water lily', 'lotus', 
        'driftwood', 'beach', 'rain', 'cucumber', 'watermelon', 'melon', 
        'sea breeze', 'ozone', 'fresh water', 'marine', 
    ],
    'Gourmand': [
        'coffee', 'almond', 'coconut', 'praline', 'licorice', 'chocolate', 
        'caramel', 'vanilla', 'hazelnut', 'cinnamon', 'milk', 'cream', 
        'butter', 'bread', 'popcorn', 'honey', 'maple syrup', ''
    ]
}

# encode category example
category_embeddings = {}
for category, examples in categories.items():
    category_embeddings[category] = model.encode(examples).mean(axis=0)

# categorize each note
def categorize_note(note):
    note_embedding = model.encode([note])[0]
    similarities = {category: cosine_similarity([note_embedding], [emb])[0][0] 
                    for category, emb in category_embeddings.items()}
    return max(similarities, key=similarities.get)

categorized_notes = [(note, categorize_note(note)) for note in notes]


result_df = pd.DataFrame(categorized_notes, columns=['Note', 'Category'])
result_df.to_csv('categorized_notes.csv', index=False)

print("\nCategory Distribution:")
print(result_df['Category'].value_counts())