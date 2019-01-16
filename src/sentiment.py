import numpy as np
import tensorflow as tf
import csv
import json
from string import punctuation
from collections import Counter

file_in = "../datasets/data.csv"
with open(file_in, "r") as f_csv:
    reviews = []
    labels = []
    ce = csv.reader(f_csv)
    rows = [row for row in ce]
    #reviews = ce[:,0]
    #lables = ce[:,1]
    for line in rows:
    #    line = line.split(',')
    #    print(line)
        reviews.append(line[0])
        labels.append(line[1])

all_text = []

for review in reviews:
    tmp = ''.join([c for c in review if (c not in punctuation)])
    tmp = tmp.replace("\n", "")
    all_text.append(tmp)


#all_text = ''.join(reviews)
words = []
for i in range(len(all_text)):
    words.append(all_text[i].split())
print(all_text)

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word : ii for ii, word in enumerate(vocab, 1)}
reviews_ints = []

#for review in reviews:
#    reviews_ints.append([vocab_to_int[word] for word in review.split()])
#print(len(reviews_ints))
#print(reviews_ints[1])
