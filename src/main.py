# encoding:utf-8

import numpy as np
import csv
wordsList = np.load('wordsList.npy')
print('载入word列表')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8')
             for word in wordsList]
wordVectors = np.load('wordVectors.npy')
print('载入文本向量')

print(len(wordsList))
print(wordVectors.shape)

import os
from os.path import isfile, join


file_in = "../datasets/data.csv"
with open(file_in, "r") as f_csv:
    reviews = []
    review_labels = []
    ce = csv.reader(f_csv)
    rows = [row for row in ce]
    #reviews = ce[:,0]
    #lables = ce[:,1]
    for line in rows:
    #    line = line.split(',')
    #    print(line)
        reviews.append(line[0])
        review_labels.append(line[1])

counter = []
for i in range(len(review_labels)):
    counter.append(len(reviews[i].split()))

num_files = len(reviews)


print('文件总数', num_files)
print('所有的词的数量', sum(counter))
print('平均文件词的长度', sum(counter) / len(reviews))

import re
max_seq_num = 150

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
num_dimensions = 300  # Dimensions for each word vector


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())




ids = np.zeros((num_files, max_seq_num), dtype='int32')
file_count = 0
for line in reviews:
    indexCounter = 0
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
        try:
            ids[file_count][indexCounter] = wordsList.index(word)
        except ValueError:
            ids[file_count][indexCounter] = 399999  # 未知的词
        indexCounter = indexCounter + 1
        if indexCounter >= max_seq_num:
            break
    file_count = file_count + 1
    if (file_count % 200 == 0):
        print(file_count)


np.save('idsMatrix', ids)
print("save idxMatrix done..")

from random import randint

batch_size = 2000
lstm_units = 64
num_labels = 2
iterations = 500
lr = 0.001
ids = np.load('idsMatrix.npy')


def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(1, 2500)
        if (review_labels[num] == 1):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num]
        #if (i % 2 == 0):
        #    num = randint(1, 1999)
        #    labels.append([1, 0])
        #else:
        #    num = randint(2999, 5000)
        #    labels.append([0, 1])
        #arr[i] = ids[num - 1:num]
        #arr[i] = ids[num]
    return arr, labels


def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    #print(arr.shape)
    for i in range(batch_size):
        num = randint(2500, 5000)
        if (review_labels[num] == 1):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        #arr[num] = ids[num]
        #num = randint(1999, 2999)
        #if (num <= 2500):
        #    labels.append([1, 0])
        #else:
        #    labels.append([0, 1])
        #print(arr[i].shape)
        #print(num)
        #print(ids.shape)
        #print(ids[num].shape)
        #print(num)
        arr[i] = ids[num]
        #arr[i] = ids[num - 1:num]
    return arr, labels


import tensorflow as tf

tf.reset_default_graph()
print("tf.reset done...")
labels = tf.placeholder(tf.float32, [batch_size, num_labels])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])
data = tf.Variable(
    tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)
print("data done...")
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.5)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
print("value done...")
weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))
bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)
print("prediction done...")
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

saver = tf.train.Saver()
print("saver done...")
with tf.Session() as sess:
    if os.path.exists("models_test") and os.path.exists("models_test/checkpoint"):
        saver.restore(sess, tf.train.latest_checkpoint('models'))
    else:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
    print("run done...")
    iterations = 500
    for step in range(iterations):
        next_batch, next_batch_labels = get_test_batch()
        if step % 20 == 0:
            print("step:", step, " 正确率:", (sess.run(
                accuracy, {input_data: next_batch, labels: next_batch_labels})) * 100)

    if not os.path.exists("models"):
        os.mkdir("models")
    save_path = saver.save(sess, "models/model.ckpt")
    print("Model saved in path: %s" % save_path)
