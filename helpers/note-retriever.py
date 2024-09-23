import pandas as pd
import ast

# import cleaned excel file
perfume_data = pd.read_excel('../cleaned_perfume_data.xlsx')

# convert list representations into list
perfume_data['notes'] = perfume_data['notes'].apply(ast.literal_eval)

# normalize and clean up notes
def clean_note(note):
    # split notes by comma
    return [part.strip().lower() for part in note.split(',')]

# flatten notes column into a single list, clean each note, and handle commas
all_notes = [cleaned_note for sublist in perfume_data['notes'] for note in sublist for cleaned_note in clean_note(note)]

# remove duplicates
unique_notes = sorted(set(all_notes))

# save notes into txt
with open('unique_notes_cleaned.txt', 'w') as f:
    for note in unique_notes:
        f.write(f"{note}\n")

print("Cleaned unique notes have been saved to 'unique_notes_cleaned.txt'")