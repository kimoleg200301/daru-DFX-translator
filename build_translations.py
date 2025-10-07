import csv
import json

with open('translations.json', encoding='utf-8') as f:
    translations = json.load(f)

# Load entries to ensure coverage
with open('extracted_texts.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

missing = [r['text_en'] for r in rows if r['text_en'] not in translations]
if missing:
    raise SystemExit(f"Missing translations: {len(missing)} entries")

for key, value in translations.items():
    if not value:
        raise SystemExit(f"Empty translation for: {key!r}")

# Write map.csv
with open('map.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text_en', 'text_ru'])
    for row in rows:
        writer.writerow([row['text_en'], translations[row['text_en']]])

# Update extracted_texts.csv with Russian column
for row in rows:
    row['text_ru'] = translations[row['text_en']]

with open('extracted_texts.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['count', 'text_en', 'text_ru']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print('Updated map.csv and extracted_texts.csv')
