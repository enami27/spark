import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load pretrained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load cat notes
categorized_notes_df = pd.read_csv('../categorized_notes.csv')

# load fragrances
perfumes_df = pd.read_excel('../cleaned_perfume_data.xlsx') 

# reconstruct categories from the CSV
# each category from the unique notes file is used as a key to store all notes in that category as a list

categories = {}
for category in categorized_notes_df['Category'].unique():
    categories[category] = categorized_notes_df[categorized_notes_df['Category'] == category]['Note'].tolist()

# encode category examples : an embedding is generated for each note
# The mean of all embeddings in a category is taken to represent the category

category_embeddings = {}
for category, notes in categories.items():
    category_embeddings[category] = model.encode(notes).mean(axis=0)

# categorize each fragrance
def categorize_fragrance(notes):
    # combine all notes
    combined_notes = ' '.join(notes)
    
    # encode combined notes
    fragrance_embedding = model.encode([combined_notes])[0]
    
    # comparing each fragrance to category by calculating cosine similarity between the fragrance embedding and each category embedding

    similarities = {category: cosine_similarity([fragrance_embedding], [emb])[0][0] 
                    for category, emb in category_embeddings.items()}
    
    # get top 3 categories
    top_categories = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return top_categories

# categorize each fragrance
perfumes_df['Top Categories'] = perfumes_df['notes'].apply(lambda x: categorize_fragrance(x.split(',')))

# save results
perfumes_df.to_excel('categorized_perfumes.xlsx', index=False)

print("Fragrance categorization complete. Results saved to 'categorized_perfumes.xlsx'")