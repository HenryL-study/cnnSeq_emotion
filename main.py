# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import codecs
from dataloader import Dataloader
from cnnSeq_model import Model

#########################################################################################
#  Model  Hyper-parameters
######################################################################################
HIDDEN_DIM = 128 # hidden state dimension of lstm cell
SEQ_LENGTH = 50 # sequence length TODO need processing data
START_TOKEN = 1 # no use
PRE_EPOCH_NUM = 60 # supervise (maximum likelihood estimation) epochs
BATCH_SIZE = 64
labels = 2
gen_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
gen_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
TOTAL_BATCH = 1000 #TODO
SEED = 88

sess = tf.InteractiveSession()

#Parameters
src_vocab_size = None
embedding_size = None
glove_embedding_filename = 'data/glove-vec.npy'
positive_file = 'data/pos-vec.txt'
positive_len_file = 'data/pos-len.txt' 
negative_file = 'data/neg-vec.txt'
negative_len_file = 'data/neg-len.txt'

#Word embedding
def loadGloVe(filename):
    embd = np.load(filename)
    return embd

random.seed(SEED)
np.random.seed(SEED)

#Word embedding parameters
embedding = loadGloVe(glove_embedding_filename)
embedding_size = embedding.shape[1]
src_vocab_size = embedding.shape[0]

print('Glove vector loaded. Total vocab: ', src_vocab_size, '. embedding_size: ', embedding_size)

emotion_model = Model(src_vocab_size, BATCH_SIZE, embedding_size, HIDDEN_DIM, embedding, SEQ_LENGTH, START_TOKEN, gen_filter_sizes, gen_num_filters, labels)
dataloader = Dataloader(BATCH_SIZE)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    supervised_g_test_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.total_batch):
        x_batch, x_batch_len, y_batch = data_loader.next_batch()
        print(y_batch)
        loss, _, sample = emotion_model.train_step(sess, x_batch, x_batch_len, y_batch)
        # print("sample shape: ", sample[0])
        supervised_g_losses.append(loss)

    for it in range(data_loader.num_test_batch):
        x_batch, x_batch_len, y_batch = data_loader.next_test_batch()
        test_loss, sample = emotion_model.test_step(sess, x_batch, x_batch_len, y_batch)
        # print("sample shape: ", sample[0])
        supervised_g_test_losses.append(test_loss)

    return np.mean(supervised_g_losses), np.mean(supervised_g_test_losses), sample

#  pre-train generator
print ('Start training...')
dataloader.load_train_data(positive_file, positive_len_file, negative_file, negative_len_file)
for epoch in range(PRE_EPOCH_NUM):
    loss, test_loss, sample = train_epoch(sess, emotion_model, dataloader)
    if epoch % 1 == 0:
        print ('train epoch ', epoch, 'generator_loss ', loss, 'test_loss ', test_loss)




