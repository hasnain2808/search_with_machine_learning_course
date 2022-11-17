import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
print(queries_df[queries_df['query'] == 'Beats By Dr. Dre- Monster Pro Over-the-Ear Headphones -' ])

queries_df['query'] = queries_df['query'].str.lower()
print(queries_df['query'])
queries_df['query'] = queries_df['query'].str.replace('[^a-z0-9\w]', ' ', regex=True)
print(queries_df['query'])
queries_df['query'] = queries_df['query'].str.replace('\s+', ' ', regex=True)


# regexp = RegexpTokenizer('\w+')
# queries_df['query'] = queries_df['query'].apply(regexp.tokenize)

print(queries_df['query'])
queries_df['query'] = queries_df['query'].str.split()
print(queries_df['query'])

stemmer = PorterStemmer()
queries_df['query'] = queries_df['query'].apply( lambda word_list: ' '.join([stemmer.stem(word) for word in word_list ]))




# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
while True:
    
    cats_to_replace = queries_df[queries_df['category'].map(queries_df['category'].value_counts()) < 10001]['category'].tolist()
    cats_to_replace = set(cats_to_replace)
    print(len(cats_to_replace))
    if not cats_to_replace:
        break
    
    for category in tqdm(cats_to_replace):
        # print(category)
        if category == 'cat00000':
            continue

        parent = parents_df[parents_df['category'] ==  category ]['parent'].iloc[0]

        queries_df.loc[queries_df['category'] == category, 'category'] = parent

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)


# commands 
# ~/fastText-0.9.2/fasttext supervised -input  /workspace/datasets/fasttext/train.txt -output /workspace/datasets/fasttext/model -lr 0.5 -epoch 25
# ~/fastText-0.9.2/fasttext test  /workspace/datasets/fasttext/model.bin  /workspace/datasets/fasttext/test.txt 5