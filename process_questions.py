#-*- coding:utf-8 -*-
from __future__ import print_function
import os
import codecs
import re
import tensorflow
import numpy as np



from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#Parameters
embedding_size = 25
glove_embedding_filename = 'glove.twitter.27B.25d.txt'
pos_filename = 'data/aaa.txt' #'question-simple.txt'
neg_filename = 'data/bbb.txt'

processed_pos_filename = 'data/pos-vec.txt'
processed_pos_ques_len = 'data/pos-len.txt'
processed_neg_filename = 'data/neg-vec.txt'
processed_neg_ques_len = 'data/neg-len.txt'
processed_glove = 'data/glove-vec'
index_to_word = 'data/index_to_word.txt'


pos_ques = []
MAX_LENGTH = 0
file = open(pos_filename,'r')
for line in file.readlines():
    row = 'starttrats ' + line.strip() + ' enddne'
    row_ = text_to_word_sequence(row)
    MAX_LENGTH = max(MAX_LENGTH, len(row_))
    pos_ques.append(row)
file.close()

neg_ques = []
file = open(neg_filename,'r')
for line in file.readlines():
    row = 'starttrats ' + line.strip() + ' enddne'
    row_ = text_to_word_sequence(row)
    MAX_LENGTH = max(MAX_LENGTH, len(row_))
    neg_ques.append(row)
file.close()

embedding_index = {}
fopen = codecs.open(glove_embedding_filename, 'r', 'utf-8')
i=0
for eachLine in fopen.readlines():
    # First element in each line is the word
    values = eachLine.split()
    if len(values) < 2:
        print(i)
    word = values[0]
    # Word vectors
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
    i+=1
fopen.close()
embedding_index['starttrats'] = np.asarray(['0' for _ in range(embedding_size)], dtype='float32')
embedding_index['enddne'] = np.asarray(['0' for _ in range(embedding_size)], dtype='float32')

print('Found %s word vectors.' % len(embedding_index))

ques = pos_ques + neg_ques
tokenizer = Tokenizer()
tokenizer.fit_on_texts(ques)
pos_sequences = tokenizer.texts_to_sequences(pos_ques)
neg_sequences = tokenizer.texts_to_sequences(neg_ques)

ques_len_static = [0,0,0,0,0,0,0]

pos_ques_len = codecs.open(processed_pos_ques_len,'w', 'utf-8')
for seq in pos_sequences:
    if len(seq) < 50: 
        ques_len_static[0] += 1
        pos_ques_len.write(str(len(seq)))
        pos_ques_len.write(" ")
    elif len(seq) < 100:
        ques_len_static[1] += 1
        pos_ques_len.write(str(len(seq)))
        pos_ques_len.write(" ")
    elif len(seq) < 200:
        ques_len_static[2] += 1
        pos_ques_len.write("100")
        pos_ques_len.write(" ")
    elif len(seq) < 300:
        ques_len_static[3] += 1
        pos_ques_len.write("100")
        pos_ques_len.write(" ")
    elif len(seq) < 400:
        ques_len_static[4] += 1
        pos_ques_len.write("100")
        pos_ques_len.write(" ")
    elif len(seq) < 500:
        ques_len_static[5] += 1
        pos_ques_len.write("100")
        pos_ques_len.write(" ")
    else:
        ques_len_static[6] += 1
        pos_ques_len.write("100")
        pos_ques_len.write(" ")
pos_ques_len.close()

neg_ques_len = codecs.open(processed_neg_ques_len,'w', 'utf-8')
for seq in pos_sequences:
    if len(seq) < 50: 
        ques_len_static[0] += 1
        neg_ques_len.write(str(len(seq)))
        neg_ques_len.write(" ")
    elif len(seq) < 100:
        ques_len_static[1] += 1
        neg_ques_len.write(str(len(seq)))
        neg_ques_len.write(" ")
    elif len(seq) < 200:
        ques_len_static[2] += 1
        neg_ques_len.write("100")
        neg_ques_len.write(" ")
    elif len(seq) < 300:
        ques_len_static[3] += 1
        neg_ques_len.write("100")
        neg_ques_len.write(" ")
    elif len(seq) < 400:
        ques_len_static[4] += 1
        neg_ques_len.write("100")
        neg_ques_len.write(" ")
    elif len(seq) < 500:
        ques_len_static[5] += 1
        neg_ques_len.write("100")
        neg_ques_len.write(" ")
    else:
        ques_len_static[6] += 1
        neg_ques_len.write("100")
        neg_ques_len.write(" ")
neg_ques_len.close()

print("ques_len_static:\n", ques_len_static)
#print(ques[0])
#print(sequences_train[0])
# # Auto filled with 0
# remove MAX_LENGTH setting below to use the max length of all sentences.
MAX_LENGTH = 100
pos_data_train = pad_sequences(pos_sequences, maxlen = MAX_LENGTH, padding='post', truncating='post')
neg_data_train = pad_sequences(neg_sequences, maxlen = MAX_LENGTH, padding='post', truncating='post')


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Prepare embedding matrix
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words, embedding_size))
in_to_word = {}
for word, i in word_index.items():
    #print(word)
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
    in_to_word[i] = word

in_w = codecs.open(index_to_word,'w', 'utf-8')
for i, word in in_to_word.items():
    in_w.write(str(i) + ' ' + unicode(word, 'utf8')+'\n')
in_w.close()

np.save(processed_glove,embedding_matrix)
np.savetxt(processed_pos_filename,pos_data_train, fmt="%d", delimiter=' ')
np.savetxt(processed_neg_filename,neg_data_train, fmt="%d", delimiter=' ')

print("Processing done.")
print("Max length: ", MAX_LENGTH)
print("Embedding shape: ", embedding_matrix.shape)
print("Data shape: ", data_train.shape)


# #Word embedding
# def loadGloVe(filename):
#     vocab = []
#     embd = []
#     vocab.append('unk') #装载不认识的词
#     embd.append([0]*embedding_size) #这个emb_size可能需要指定
#     file = codecs.open(filename, 'r', 'utf-8')
#     for line in file.readlines():
#         row = line.strip().split(' ')
#         vocab.append(row[0])
#         embd.append(row[1:])
#     print('GloVe loaded.')
#     file.close()
#     return vocab,embd


# vocab,embd = loadGloVe(glove_embedding_filename)
# embedding_size = len(embd[0])
# #Add start & end & unknown & pad token
# PAD_TOKEN = 0
# vocab.insert(0, '<p_a_d>')
# embd.insert(0, ['0' for _ in range(embedding_size)])
# START_TOKEN = len(vocab)
# vocab.append('<s_t_a_r_t>')
# embd.append(['0' for _ in range(embedding_size)])
# END_TOKEN = len(vocab)
# vocab.append('<e_n_d>')
# embd.append(['0' for _ in range(embedding_size)])
# UKNOWN_TOKEN = len(vocab)
# vocab.append('<u_k_n_o_w_n>')
# embd.append(['0' for _ in range(embedding_size)])
# src_vocab_size = len(vocab)

# #vocab to int
# vocab_to_int = {}
# for i in range(src_vocab_size):
#     vocab_to_int[vocab[i]] = i

# print('Glove vector loaded. Total vocab: ', src_vocab_size, '. embedding_size: ', embedding_size)

# ques = []
# MAX_LENGTH = 0
# file = codecs.open(question_filename,'r', 'utf-8')
# for line in file.readlines():
#     row = line.strip()
#     row = text_to_word_sequence(row)
#     MAX_LENGTH = max(MAX_LENGTH, len(row))
#     ques.append(row)
# file.close()

# #to int & reconstruct embedding
# re-embed = []
# re-vocab = []
# for q in ques:
#     for word in q:
        
# embedding = np.asarray(embd)

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(ques)

# sequences_ques = tokenizer.texts_to_sequences(ques)
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
# sequences_ques = pad_sequences(sequences_ques)
# print('questions shape: ', sequences_ques.shape)

# # Prepare embedding matrix
# num_words = len(word_index) + 4
# embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i == 0:
#         print('impossible!!!')
#     if word in vocab:
#         embedding_vector = embedding[vocab.index(word)]
#         embedding_matrix[i] = embedding_vector
