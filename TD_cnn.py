#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import pickle
import h5py
import os
import numpy as np
np.random.seed(1337)  # for reproducibility
from scipy import spatial
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Merge, Convolution1D, MaxPooling1D, LSTM, merge,  Reshape, Lambda, AveragePooling1D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot
import theano.tensor as T
from keras.engine.topology import Layer, InputSpec

def get_activations(gan, layer, X_batch):
    get_activations = K.function([gan.layers[0].input, K.learning_phase()], [gan.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

#list of paremeter 
nb_filters = 50
filters_size = 5
sentene_word_num = 60
document_word_num = 620
word_vector_dim = 50
batch_size = 20
nb_epoch = 5

answer_files = [answers_text_form for answers_text_form in os.listdir('./Sentence_Segment_Human_TD_aWid/raw_word_id')]

########## Read file ##########

print ('Load data...')

with h5py.File('COMPRESS_TD_LOAD_to_TRAIN_ALL.h5', 'r') as hf:
	print('List of arrays in this file: \n', hf.keys())
	sentence_train_two_dim_form = hf.get('sentence_train_two_dim_form')
	sentence_dev_two_dim_form = hf.get('sentence_dev_two_dim_form')
	sentence_test_two_dim_form = hf.get('sentence_test_two_dim_form')
	document_train_two_dim_form = hf.get('document_train_two_dim_form')
	document_dev_two_dim_form = hf.get('document_dev_two_dim_form')
	document_test_two_dim_form = hf.get('document_test_two_dim_form')
	answer_train_two_dim_form = hf.get('answer_train_two_dim_form')
	answer_dev_two_dim_form = hf.get('answer_dev_two_dim_form')
	answer_test_two_dim_form = hf.get('answer_test_two_dim_form')

	#to numpy array and reshape
	sentence_train_two_dim_form = np.asarray(sentence_train_two_dim_form)
	sentence_dev_two_dim_form = np.asarray(sentence_dev_two_dim_form)
	sentence_test_two_dim_form = np.asarray(sentence_test_two_dim_form)

	document_train_two_dim_form = np.asarray(document_train_two_dim_form)
	document_dev_two_dim_form = np.asarray(document_dev_two_dim_form)
	document_test_two_dim_form = np.asarray(document_test_two_dim_form)

	answer_train_two_dim_form = np.asarray(answer_train_two_dim_form)
	answer_dev_two_dim_form = np.asarray(answer_dev_two_dim_form)
	answer_test_two_dim_form = np.asarray(answer_test_two_dim_form)

print(sentence_train_two_dim_form.shape)
############ Build model ############

print ('build model')

sentence_vec = Input(shape=(sentene_word_num,word_vector_dim), name='sen_input')
document_vec = Input(shape=(document_word_num,word_vector_dim), name='doc_input')
conv_sen = Convolution1D(nb_filters, filters_size, border_mode='same',  activation='sigmoid', name='sen_cov')
conv_doc = Convolution1D(nb_filters, filters_size, border_mode='same',  activation='sigmoid', name='doc_cov')

conved_sen = conv_sen(sentence_vec)
conved_doc = conv_doc(document_vec)

pool_sen = MaxPooling1D(pool_length=sentene_word_num, stride=None, border_mode='valid', name='sen_pool')
pool_doc = MaxPooling1D(pool_length=document_word_num, stride=None, border_mode='valid', name='doc_pool')

pooled_sen = pool_sen(conved_sen)
pooled_doc = pool_doc(conved_doc)

flaten_sen = Flatten()(pooled_sen)
flaten_doc = Flatten()(pooled_doc)

#sim matrix
sim_matrix = Dense(nb_filters, bias=None, name='sim_matrix')(flaten_sen)

#in dot form
dot_similarity = merge([sim_matrix, flaten_doc], mode='dot', dot_axes=1, name='dot_similarity')
dot_similarity = Reshape((1,))(dot_similarity)


inter_mid = merge([flaten_sen, flaten_doc, dot_similarity], mode='concat', name='intermid_vec')

x = Dense(251, activation='tanh', name='hidden_layer1')(inter_mid)
#x = Dropout(0.5)(x)

output = Dense(1, activation='sigmoid', name='logistic_regression')(x)
model = Model(input=[sentence_vec, document_vec], output=output)


# Draw the structure of model
plot(model, to_file='./model_pic.png')

model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

model.save_weights('TD_cnn_before_train_weights.h5', overwrite=True)
#model.load_weights('5.0_sim_after_train_weights.h5')
model.summary()

model.fit([sentence_train_two_dim_form[None::], document_train_two_dim_form[None::]], answer_train_two_dim_form,
 batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.0, 
 validation_data=([sentence_test_two_dim_form[None::],document_test_two_dim_form[None::]],answer_test_two_dim_form), 
 shuffle=True, class_weight=None, sample_weight=None)

score = model.evaluate([sentence_test_two_dim_form[None::],document_test_two_dim_form[None::]], answer_test_two_dim_form, verbose=0)

print ('score:'+str(score))

###這個是模型的預測值###
predict_list = model.predict([sentence_test_two_dim_form[None::],document_test_two_dim_form[None::]], batch_size= batch_size, verbose=0)

json_string = model.to_json()

model.save_weights('TD_cnn_after_train_weights.h5', overwrite=True)
