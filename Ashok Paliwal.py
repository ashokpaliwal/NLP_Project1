#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from os import listdir
from os.path import isfile, join
import json
import pandas as pd
 
path='/users/ashokpaliwal/Desktop/NLP_Project1/data'
 
# Processing JSON files
json_dir = path + '/docs'
json_files = [f for f in listdir(json_dir) if isfile(join(json_dir, f))]
input_data = []
for i in range(len(json_files)):
    file = json_dir + '/' + json_files[i]
    with open(file) as f:  
        data = json.load(f)
        doc_info = [data["_id"], data["jd_information"]["description"]]
        input_data.append(doc_info)
 
json_data = pd.DataFrame(input_data)
json_data.columns = ['Document ID', 'JD']
json_data['Document ID'] = json_data['Document ID'].astype('int64')
json_data['JD'] = json_data['JD'].astype(str)
 
# Importing CSV file
depts = pd.read_csv(path + '/' + 'document_departments.csv')
 
# Joining CSV and JSON datasets to form training dataset
full_data = pd.merge(json_data, depts, on = 'Document ID', how = 'left')
 
# Testing empty JDs
#j=1
#for i in range(len(full_data['JD'])):
#    JD = full_data['JD'][i]
#    if JD == '': 
#        print(j, i)
#        j = j + 1
 
#TBD - Remove rows with emmpty JDs
full_data = full_data[full_data['JD'] != '']
#Randomizing dataset for test and train separation
full_data = full_data.sample(frac=1).reset_index(drop=True)
 
import tensorflow as tf
from tensorflow import keras
import numpy as np
 
train_data = list(full_data['JD'].values)
t = keras.preprocessing.text.Tokenizer(num_words=10000)
t.fit_on_texts(train_data)
vocab_size = len(t.word_index) + 1
encoded_data = t.texts_to_sequences(train_data)
max_doc_len = len(max(encoded_data, key=len)) + 1
padded_data = keras.preprocessing.sequence.pad_sequences(
        encoded_data, maxlen = max_doc_len, padding='post')
 
labels = list(full_data['Department'].values)
unique_labels = list(set(labels))
label_indexes = dict()
for i in range(len(unique_labels)):
    label_indexes[i] = unique_labels[i]
 
encoded_labels = []
for label in labels:
    for val, word in label_indexes.items():
        if word == label:
            encoded_labels.append(val)
 
#Import GloVe mappings
path2='/users/ashokpaliwal/Desktop/NLP_Project1/glove.6B'
embeddings_index = dict()
f = open(path2 + '/' + 'glove.6B.50d.txt', encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
 
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
 
#Build the model
e = keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], 
                           input_length=max_doc_len, trainable=False)
model = keras.Sequential()
model.add(e)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation=tf.nn.relu))
model.add(keras.layers.Dense(len(unique_labels), activation=tf.nn.softmax))
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
 
partial_train_data = padded_data[100:700]
partial_train_labels = encoded_labels[100:700]
 
partial_test_data = padded_data[:100]
partial_test_labels = encoded_labels[:100]
 
x_val = padded_data[700:]
y_val = encoded_labels[700:]
 
history = model.fit(partial_train_data, partial_train_labels, epochs=30, 
                    batch_size = 512, validation_data = (x_val, y_val), verbose=1)
# evaluate the model
results = model.evaluate(partial_test_data, partial_test_labels, verbose=0)
print(results)

