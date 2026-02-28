"""
Script to replace the data analytics entry in technical_kb.csv with enhanced version
"""
import pandas as pd
import csv

# Read the enhanced analytics
enhanced_df = pd.read_csv('data/analytics_enhanced.csv')
enhanced_answer = enhanced_df.iloc[0]['answer']

# Read the current technical KB
with open('data/technical_kb.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

# Find and replace the data analytics entry
for i, row in enumerate(rows):
    if len(row) >= 2 and 'What data is collected and how is it analyzed?' in row[0]:
        print(f"Found data analytics entry at row {i}")
        print(f"Old answer length: {len(row[1])} characters")
        print(f"New answer length: {len(enhanced_answer)} characters")
        rows[i] = [row[0], enhanced_answer]
        break

# Write back to file
with open('data/technical_kb.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("✅ Successfully replaced data analytics entry!")
print(f"📊 New analytics dashboard has {len(enhanced_answer)} characters")
