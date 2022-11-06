import fasttext
model = fasttext.load_model('/workspace/datasets/fasttext/normalized_title_model.bin')
file = open('/workspace/datasets/fasttext/top_words.txt', 'r')
synonyms = open("/workspace/datasets/fasttext/synonyms.csv", 'w')

threshold = 0.75
for word in file.readlines():
    word = word.replace('\n', '')
    for x in list(filter(lambda x: x[0] >= threshold, model.get_nearest_neighbors(word))):
        neighbors = x[1]
    if len(neighbors) > 0:
        synonyms.write(word + ',' + ','.join(neighbors) + '\n')