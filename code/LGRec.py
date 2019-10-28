# coding:utf8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import cPickle as pickle
import math
import heapq
import random

def read_data(path):

    user_num = 0
    item_num = 0
    user_length = 0
    item_length = 0

    #load all train ratings
    train_ratings = []
    with open(path + '.train.rating') as infile:
        line = infile.readline()
        while line != None and line != "":
            u, v = line.strip().split('\t')[:2]
            user_num = max(user_num, int(u))
            item_num = max(item_num, int(v))
            train_ratings.append([int(u), int(v)])
            line = infile.readline()
    
    #load ratings as map
    user_map_item = {}
    item_map_user = {}
    for u, v in train_ratings:
        if u not in user_map_item:
            user_map_item[u] = {}
        if v not in item_map_user:
            item_map_user[v] = {}
        user_map_item[u][v] = 1
        item_map_user[v][u] = 1

    for u in user_map_item:
        user_length = max(user_length, len(user_map_item[u].keys()))
    for v in item_map_user:
        item_length = max(item_length, len(item_map_user[v].keys()))

    #load all test ratings
    test_ratings = []
    with open(path + '.test.rating') as infile:
        line = infile.readline()
        while line != None and line != "":
            arr = line.strip().split('\t')
            u, v = int(arr[0]), int(arr[1])
            user_num = max(user_num, u)
            item_num = max(item_num, v)
            test_ratings.append([u, v])
            line = infile.readline()

    #load negative sample
    negative_ratings = []
    with open(path + '.test.negative') as infile:
        line  = infile.readline()
        while line != None and line != "":
            arr = line.strip().split('\t')
            negs = []
            for x in arr[1:]:
                negs.append(int(x))
            negative_ratings.append(negs)
            line = infile.readline()

    user_neighbor = {}
    item_neighbor = {}
    user_length = 0
    with open(path + '.user_neighbor_100') as infile:
        line = infile.readline()
        while line != None and line != "":
            arr = line.strip().split('\t')
            u = int(arr[0])
            v = int(arr[1])
            if u not in user_neighbor:
                user_neighbor[u] = []
            user_neighbor[u].append(v)
            user_length = max(user_length, len(user_neighbor[u]))
            line = infile.readline()

    item_length = 0
    with open(path + '.item_neighbor_100') as infile:
        line = infile.readline()
        while line != None and line != "":
            arr = line.strip().split('\t')
            v = int(arr[0])
            u = int(arr[1])
            if v not in item_neighbor:
                item_neighbor[v] = []
            item_neighbor[v].append(u)
            item_length = max(item_length, len(item_neighbor[v]))
            line = infile.readline()


    relation_map_label = {}
    with open(path + '.label') as infile:
        for line in infile.readlines():
            arr = line.strip().split(' ')
            u, m = arr[0].strip().split(',')
            u = int(u)
            m = int(m)
            if u not in relation_map_label:
                relation_map_label[u] = {}
            relation_map_label[u][m] = []
            for label in arr[1:]:
                relation_map_label[u][m].append(int(label))

    user_pop = [0] * user_num
    item_pop = [0] * item_num
    for u in user_map_item:
        user_pop[u - 1] = len(user_map_item[u].keys())
    for v in item_map_user:
        item_pop[v - 1] = len(item_map_user[v].keys())

    return user_num, item_num, user_length, item_length, train_ratings, user_map_item, item_map_user, \
           test_ratings, negative_ratings, user_pop, item_pop, user_neighbor, item_neighbor, relation_map_label

class Config(object):
    def __init__(self):
        self.path = '../data/ml-100k'
        self.batch_size = 256       # The batch size
        self.embedding_dim = 256    # The dimension of embedding layers
        self.hidden_dim = 256       # The dimension of hidden layer of dense layers
        self.user_num = 0           # The total number of items
        self.item_num = 0           # The total number of users
        self.user_length = 0        # The number of users' neighbors
        self.item_length = 0        # The number of items' neighbors
        self.label_size = 4         # The number of labels for multi-label classification  
        self.margin = 2.0           # Margin; Refer to Eq.(12)
        self.epochs = 100           # Max training epochs
        self.topK = 10              # The length of ranklist
        self.l2 = 0.05              # Regurlarization for dense layer
        self.dropout_rate = 1.0     # Dropout rate; Useless parameter
        self.alpha = 0.1            # Refer to Eq.(13) in paper
        self.beta = 0.01            # Refer to Eq.(13) in paper
        self.distance = 1           # 1: l1 normalization for Eq.(11) in paper; 2: l2 normalization for Eq.(11) in papers;
        self.learn_rate = 0.001     # Learning rate

class pRankModel(object):
    def __init__(self, config):
        batch_size = config.batch_size
        embedding_dim = config.embedding_dim
        user_num = config.user_num
        item_num = config.item_num
        user_length = config.user_length
        item_length = config.item_length
        margin = config.margin
        l2 = config.l2
        hidden_dim = config.hidden_dim
        dropout_rate = config.dropout_rate
        label_size = config.label_size
        alpha = config.alpha
        beta = config.beta
        distance = config.distance

        print 'Model config : '
        print 'Batch size = ', batch_size
        print 'Embedding dim = ', embedding_dim
        print 'Hidden dim = ', hidden_dim
        print 'Margin = ', margin
        print 'l2 regularizer = ', l2
        print 'Dropout rate = ', dropout_rate
        print 'alpha = ', alpha
        print 'beta = ', beta

        regularizer = tf.contrib.layers.l2_regularizer(l2)

        self.pos_u = tf.placeholder(tf.int32, [None, user_length])
        self.pos_v = tf.placeholder(tf.int32, [None, item_length])

        self.neg_u = tf.placeholder(tf.int32, [None, user_length])
        self.neg_v = tf.placeholder(tf.int32, [None, item_length])

        self.user_embedding = tf.get_variable(name = "user_embedding", 
                                         shape = [user_num, embedding_dim],
                                         initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.item_embedding = tf.get_variable(name = "item_embedding", 
                                         shape = [item_num, embedding_dim],
                                         initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        #pos_u_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.user_embedding, self.pos_u), 2)
        pos_u_e = tf.nn.embedding_lookup(self.user_embedding, self.pos_u)
        pos_u_d = tf.layers.dense(inputs = pos_u_e,
                           units = hidden_dim,
                           activation = tf.nn.relu,
                           #use_bias = False,
                           kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                           kernel_regularizer = regularizer,
                           name = 'user_dense_1')

        pos_u_d = tf.layers.dense(inputs = pos_u_d,
                           units = hidden_dim / 2,
                           activation = tf.nn.relu,
                           #use_bias = False,
                           kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                           kernel_regularizer = regularizer,
                           name = 'user_dense_2')
                           
        #pos_v_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.item_embedding, self.pos_v), 2)
        pos_v_e = tf.nn.embedding_lookup(self.item_embedding, self.pos_v)
        pos_v_d = tf.layers.dense(inputs = pos_v_e,
                           units = hidden_dim,
                           activation = tf.nn.relu,
                           #use_bias = False,
                           kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                           kernel_regularizer = regularizer,
                           name = 'item_dense_1')
        pos_v_d = tf.layers.dense(inputs = pos_v_d,
                           units = hidden_dim / 2,
                           activation = tf.nn.relu,
                           #use_bias = False,
                           kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                           kernel_regularizer = regularizer,
                           name = 'item_dense_2')
                           
        #neg_u_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.user_embedding, self.neg_u), 2)
        neg_u_e = tf.nn.embedding_lookup(self.user_embedding, self.neg_u)
        neg_u_d = tf.layers.dense(inputs = neg_u_e,
                           units = hidden_dim,
                           activation = tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                           kernel_regularizer = regularizer,
                           name = 'user_dense_1',
                           reuse = True)
        neg_u_d = tf.layers.dense(inputs = neg_u_d,
                           units = hidden_dim / 2,
                           activation = tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                           kernel_regularizer = regularizer,
                           name = 'user_dense_2',
                           reuse = True)
        

        #neg_v_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.item_embedding, self.neg_v), 2)
        neg_v_e = tf.nn.embedding_lookup(self.item_embedding, self.neg_v)
        neg_v_d = tf.layers.dense(inputs = neg_v_e,
                           units = hidden_dim,
                           activation = tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                           kernel_regularizer = regularizer,
                           name = 'item_dense_1',
                           reuse = True)
        neg_v_d = tf.layers.dense(inputs = neg_v_d,
                           units = hidden_dim / 2,
                           activation = tf.nn.relu,
                           kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                           kernel_regularizer = regularizer,
                           name = 'item_dense_2',
                           reuse = True)

        self.w = tf.get_variable(name = 'w', 
                            shape = [hidden_dim / 2, hidden_dim / 2],
                            initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        pos_u_tmp = tf.reshape(tf.matmul(tf.reshape(pos_u_d, [-1, hidden_dim / 2]), self.w), [-1, user_length, hidden_dim / 2])
        pos_attention_matrix = tf.matmul(pos_u_tmp, tf.transpose(pos_v_d, perm = [0, 2, 1]))
        #pos_attention_matrix = tf.tanh(pos_attention_matrix)
        neg_u_tmp = tf.reshape(tf.matmul(tf.reshape(neg_u_d, [-1, hidden_dim / 2]), self.w), [-1, user_length, hidden_dim / 2])
        neg_attention_matrix = tf.matmul(neg_u_tmp, tf.transpose(neg_v_d, perm = [0, 2, 1]))
        #neg_attention_matrix = tf.tanh(neg_attention_matrix)

        self.pos_u_attention = tf.nn.softmax(tf.reduce_max(pos_attention_matrix, 2))
        self.pos_v_attention = tf.nn.softmax(tf.reduce_max(pos_attention_matrix, 1))

        self.neg_u_attention = tf.nn.softmax(tf.reduce_max(neg_attention_matrix, 2))
        self.neg_v_attention = tf.nn.softmax(tf.reduce_max(neg_attention_matrix, 1))

        pos_u_e_f = tf.reduce_sum(pos_u_e * tf.expand_dims(self.pos_u_attention, -1), 1)
        pos_v_e_f = tf.reduce_sum(pos_v_e * tf.expand_dims(self.pos_v_attention, -1), 1)

        neg_u_e_f = tf.reduce_sum(neg_u_e * tf.expand_dims(self.neg_u_attention, -1), 1)
        neg_v_e_f = tf.reduce_sum(neg_v_e * tf.expand_dims(self.neg_v_attention, -1), 1)

        
        #model relation
        self.pos_user_input = tf.placeholder(tf.int32, [None])
        self.pos_item_input = tf.placeholder(tf.int32, [None])
        self.pos_relation_label = tf.placeholder(tf.float32, [None, label_size])
        self.neg_user_input = tf.placeholder(tf.int32, [None])
        self.neg_item_input = tf.placeholder(tf.int32, [None])
        self.neg_relation_label = tf.placeholder(tf.float32, [None, label_size])
        
        self.W_1 = tf.get_variable(name = 'W_1', 
                                   shape = [embedding_dim * 2, embedding_dim / 2 * 3],
                                   initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.W_2 = tf.get_variable(name = 'W_2',
                                   shape = [embedding_dim / 2 * 3, embedding_dim],
                                   initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        self.W_r = tf.get_variable(name = 'W_r',
                                   shape = [embedding_dim, embedding_dim],
                                   initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        self.b_1 = tf.get_variable(name = 'b_1',
                                   shape = [embedding_dim / 2 * 3],
                                   initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.b_2 = tf.get_variable(name = 'b_2',
                                   shape = [embedding_dim],
                                   initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.b_r = tf.get_variable(name = 'b_r',
                                   shape = [embedding_dim],
                                   initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        self.W_o = tf.get_variable(name = 'W_o', 
                                   shape = [embedding_dim, label_size],
                                   initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.b_o = tf.get_variable(name = 'b_o',
                                   shape = [label_size],
                                   initializer = tf.contrib.layers.xavier_initializer(uniform = False))

        pos_relation_embedding = tf.concat([pos_u_e_f, pos_v_e_f], 1)
        neg_relation_embedding = tf.concat([neg_u_e_f, neg_v_e_f], 1)

        #pos_relation_embedding = pos_u_e_f * pos_v_e_f
        #neg_relation_embedding = neg_u_e_f * neg_v_e_f

        #hidden layer 1
        pos_relation_embedding = tf.nn.relu(tf.matmul(pos_relation_embedding, self.W_1) + self.b_1)
        pos_relation_embedding = tf.nn.dropout(pos_relation_embedding, dropout_rate)
        neg_relation_embedding = tf.nn.relu(tf.matmul(neg_relation_embedding, self.W_1) + self.b_1)
        neg_relation_embedding = tf.nn.dropout(neg_relation_embedding, dropout_rate)

        #hidden layer 2
        pos_relation_embedding = tf.nn.relu(tf.matmul(pos_relation_embedding, self.W_2) + self.b_2)
        pos_relation_embedding = tf.nn.dropout(pos_relation_embedding, dropout_rate)
        neg_relation_embedding = tf.nn.relu(tf.matmul(neg_relation_embedding, self.W_2) + self.b_2)
        neg_relation_embedding = tf.nn.dropout(neg_relation_embedding, dropout_rate)

        #relation layer
        pos_relation_embedding = tf.nn.relu(tf.matmul(pos_relation_embedding, self.W_r) + self.b_r)
        neg_relation_embedding = tf.nn.relu(tf.matmul(neg_relation_embedding, self.W_r) + self.b_r)

        #output layer
        pos_output = tf.matmul(pos_relation_embedding, self.W_o) + self.b_o#tf.nn.sigmoid(tf.matmul(pos_relation_embedding, self.W_o) + self.b_o)
        neg_output = tf.matmul(neg_relation_embedding, self.W_o) + self.b_o#tf.nn.sigmoid(tf.matmul(neg_relation_embedding, self.W_o) + self.b_o)
        
        if distance == 2:
            pos = tf.reduce_sum((pos_u_e_f + pos_relation_embedding - pos_v_e_f) ** 2, 1, keep_dims = True)
            neg = tf.reduce_sum((neg_u_e_f + neg_relation_embedding - neg_v_e_f) ** 2, 1, keep_dims = True)
            self.prediction = pos
        elif ditance == 1:
            pos = tf.reduce_sum(abs(pos_u_e_f + pos_relation_embedding - pos_v_e_f), 1, keep_dims = True)
            neg = tf.reduce_sum(abs(neg_u_e_f + neg_relation_embedding - neg_v_e_f), 1, keep_dims = True)
            self.prediction = pos
        else:
            exit(1)

        self.trans_loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))

        self.pos_relation_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = pos_output, 
                                                                                       labels = self.pos_relation_label))
        self.neg_relation_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = neg_output, 
                                                                                       labels = self.neg_relation_label))

        self.reg_loss = tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.b_1) \
                      + tf.nn.l2_loss(self.W_2) + tf.nn.l2_loss(self.b_2) \
                      + tf.nn.l2_loss(self.W_o) + tf.nn.l2_loss(self.b_o) \
                      + tf.nn.l2_loss(self.W_r) + tf.nn.l2_loss(self.b_r)

        tf_alpha = tf.constant(alpha, dtype = np.float32)
        tf_beta = tf.constant(beta, dtype = np.float32)
        self.loss = self.trans_loss + alpha * (self.pos_relation_loss + self.neg_relation_loss) + beta * self.reg_loss



def getHit(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1.0
    return 0.0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.0

def sample_user(user_pop):
    number = np.random.randint(0, sum(user_pop) + 1)#random.uniform(0, sum(user_pop))
    print number
    current = 0
    for i, bias in enumerate(user_pop):
        current += bias
        if number < current:
            return i + 1

def sample_item(item_pop):
    number = np.random.randint(0, sum(item_pop) + 1)#random.uniform(0, sum(item_pop))
    current = 0
    for i, bias in enumerate(item_pop):
        current += bias
        if number < current:
            return i + 1

if __name__ == '__main__':
    t1 = time.time()
    model_config = Config()

    user_num, item_num, user_length, item_length, train_ratings, user_map_item, item_map_user, \
    test_ratings, negative_ratings, user_pop, item_pop, user_neighbor, item_neighbor, relation_map_label = read_data(model_config.path)
    
    print "Load data done [%.1f s]. user_num = %d, item_num = %d, user_length = %d, item_length = %d, train_rating_num = %d, test_rating_num = %d" \
    % ((time.time() - t1), user_num, item_num, user_length, item_length, len(train_ratings), len(test_ratings))
    
    model_config.user_num = user_num + 1
    model_config.item_num = item_num + 1
    model_config.user_length = user_length
    model_config.item_length = item_length

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True 
    sess = tf.Session(config = tf_config)

    print 'Learning rate = ', model_config.learn_rate

    train_model = pRankModel(config = model_config)
    train_step = tf.train.AdamOptimizer(model_config.learn_rate).minimize(train_model.loss)
    sess.run(tf.global_variables_initializer())


    batch_pos_u = np.zeros((model_config.batch_size, model_config.user_length), np.int32) - 1
    batch_pos_v = np.zeros((model_config.batch_size, model_config.item_length), np.int32) - 1
    batch_neg_u = np.zeros((model_config.batch_size, model_config.user_length), np.int32) - 1
    batch_neg_v = np.zeros((model_config.batch_size, model_config.item_length), np.int32) - 1

    batch_pos_user_input = np.zeros((model_config.batch_size), np.int32)
    batch_pos_item_input = np.zeros((model_config.batch_size), np.int32)
    batch_pos_relation_label = np.zeros((model_config.batch_size, model_config.label_size))
    batch_neg_user_input = np.zeros((model_config.batch_size), np.int32)
    batch_neg_item_input = np.zeros((model_config.batch_size), np.int32)
    batch_neg_relation_label = np.zeros((model_config.batch_size, model_config.label_size))

    batch_u_test = np.zeros((101, model_config.user_length), np.int32) - 1
    batch_v_test = np.zeros((101, model_config.item_length), np.int32) - 1
    batch_user_test = np.zeros((101), np.int32)
    batch_item_test = np.zeros((101), np.int32)
    batch_relation_test = np.zeros((101, model_config.label_size))
    

    final_hr_list = []
    final_ndcg_list = []

    for epoch in range(model_config.epochs):
        np.random.shuffle(train_ratings)
        one_epoch_loss = 0.0
        one_epoch_batchnum = 0.0
        epoch_start_time = time.time()
        #Get batch
        for index in range(len(train_ratings) / model_config.batch_size):
            train_sample_index = 0
            for pos_u, pos_v in train_ratings[index * model_config.batch_size : (index + 1) * model_config.batch_size]:
                i = 0
               
                for v_in in user_neighbor.get(pos_u, []):
                    batch_pos_u[train_sample_index][i] = v_in
                    i += 1
                i = 0
                for u_in in item_neighbor.get(pos_v, []):
                    batch_pos_v[train_sample_index][i] = u_in
                    i += 1

                batch_pos_user_input[train_sample_index] = pos_u
                batch_pos_item_input[train_sample_index] = pos_v
                if pos_u in relation_map_label and pos_v in relation_map_label[pos_u]:
                    for idx in relation_map_label[pos_u][pos_v]:
                        batch_pos_relation_label[train_sample_index][idx] = 1.0

                #sample negative (u, v)
                neg_u = np.random.randint(1, user_num + 1)
                neg_v = np.random.randint(1, item_num + 1)
                while neg_u in user_map_item and neg_v in user_map_item[neg_u]:
                    neg_u = np.random.randint(1, user_num + 1)
                    neg_v = np.random.randint(1, item_num + 1)
                #print neg_u, neg_v
                i = 0
                for v_in in user_neighbor.get(neg_u, []):
                    batch_neg_u[train_sample_index][i] = v_in
                    i += 1
                i = 0
                for u_in in item_neighbor.get(neg_v, []):
                    batch_neg_v[train_sample_index][i] = u_in
                    i += 1

                batch_neg_user_input[train_sample_index] = neg_u
                batch_neg_item_input[train_sample_index] = pos_v
                if neg_u in relation_map_label and neg_v in relation_map_label[neg_u]:
                    for idx in relation_map_label[neg_u][neg_v]:
                        batch_neg_relation_label[train_sample_index][idx] = 1.0


                train_sample_index += 1

            #exit(1)
            #print batch_pos_user_input
            #print batch_pos_item_input
            #print batch_relation_label
            #exit(1)
            feed_dict = {
                    train_model.pos_u : batch_pos_u,
                    train_model.pos_v : batch_pos_v,
                    train_model.neg_u : batch_neg_u,
                    train_model.neg_v : batch_neg_v,
                    train_model.pos_user_input : batch_pos_user_input,
                    train_model.pos_item_input : batch_pos_item_input,
                    train_model.pos_relation_label : batch_pos_relation_label,
                    train_model.neg_user_input : batch_neg_user_input,
                    train_model.neg_item_input : batch_neg_item_input,
                    train_model.neg_relation_label : batch_neg_relation_label
            }
            _, model_loss = sess.run([train_step, train_model.loss], feed_dict)
            one_epoch_loss += model_loss
            one_epoch_batchnum += 1.0

            #initialize batch
            batch_pos_u[:, :] = -1
            batch_pos_v[:, :] = -1
            batch_neg_u[:, :] = -1
            batch_neg_v[:, :] = -1
            batch_pos_user_input[:] = 0
            batch_pos_item_input[:] = 0
            batch_pos_relation_label[:, :] = 0
            batch_neg_user_input[:] = 0
            batch_neg_item_input[:] = 0
            batch_neg_relation_label[:, :] = 0

            epoch_end_time = time.time()
            if index == len(train_ratings) / model_config.batch_size - 1:
                print 'epoch %d [%.1f]:iteration average loss = %.4f' % \
                       (epoch, epoch_end_time - epoch_start_time, one_epoch_loss / one_epoch_batchnum)

                test_start_time = time.time()
                hr_list = []
                ndcg_list = []
                for idx in xrange(len(test_ratings)):
                    rating = test_ratings[idx]
                    items = negative_ratings[idx]

                    u = rating[0]
                    gtItem = rating[1]

                    items.append(gtItem)
                    #print len(items)
                    for i in range(len(items)):
                        #print i
                        j = 0
                        for v_in in user_neighbor.get(u, []):#item_knn.get(items[i], []):#user_map_item.get(u, []):
                            batch_u_test[i][j] = v_in
                            j += 1
                        
                        j = 0
                        for u_in in item_neighbor.get(items[i], []):#user_knn.get(u, []):#item_map_user.get(items[i], []):
                            batch_v_test[i][j] = u_in
                            j += 1

                        batch_user_test[i] = u
                        batch_item_test[i] = items[i]
                        if u in relation_map_label and items[i] in relation_map_label[u]:
                           for idx in relation_map_label[u][items[i]]:
                                batch_relation_test[i][idx] = 1.0


                    feed_dict = {
                            train_model.pos_u : batch_u_test,
                            train_model.pos_v : batch_v_test,
                            train_model.pos_user_input : batch_user_test,
                            train_model.pos_item_input : batch_item_test,
                            train_model.pos_relation_label : batch_relation_test
                    }

                    pred_value = sess.run([train_model.prediction], feed_dict)
                    pre_real_val = np.array(pred_value).reshape((-1))

                    #initialize batch
                    batch_u_test[:, :] = -1
                    batch_v_test[:, :] = -1
                    batch_user_test[:] = 0
                    batch_item_test[:] = 0
                    batch_relation_test[:, :] = 0

                    map_item_score = {}
                    for i in range(len(items)):
                        item = items[i]
                        map_item_score[item] = pre_real_val[i]

                    items.pop()
                    ranklist = heapq.nsmallest(model_config.topK, map_item_score, key = map_item_score.get)
                    #print gtItem, ranklist
                    hr_list.append(getHit(ranklist, gtItem))
                    ndcg_list.append(getNDCG(ranklist, gtItem))

                hr_val, ndcg_val = np.array(hr_list).mean(), np.array(ndcg_list).mean()
                final_hr_list.append(hr_val)
                final_ndcg_list.append(ndcg_val)

                test_end_time = time.time()
                print 'test [%.1f] : hr = %.4f (%.4f), ndcg = %.4f (%.4f)' % \
                       ((test_end_time - test_start_time), hr_val, max(final_hr_list), ndcg_val, max(final_ndcg_list))

    print 'End! hr = %.4f, ndcg = %.4f' % \
            (max(final_hr_list), max(final_ndcg_list)) 

