from keras.callbacks import ModelCheckpoint, ProgbarLogger
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from utility import load_corpus,get_volcabulary_and_list_words,get_list_words

class Sentiment():
    def __init__(self,start=1, max_words=5000, max_length=50,
                 index_from=3, embedding_dims=50,oov=2):
        self.max_words = max_words
        self.max_length = max_length
        self.index_from = index_from
        self.embedding_dims = embedding_dims
        self.oov = oov
        self.start = start
        self.volcabulary=None

    def set_volcabulary(self,volcabulary):
        self.volcabulary =  sorted(volcabulary.items(), key=lambda x: x[1], reverse=True)[:self.max_words]

    def get_word_index(self,words):

        word2index = {word[0]: i for i, word in enumerate(self.volcabulary)}
        words_index = [[self.start] + [(word2index[w] + self.index_from) if w in word2index else self.oov for w in review] for
                               review in words]
        # in word2vec embedding, use (i < max_words + index_from) because we need the exact index for each word, in order to map it to its vector. And then its max_words is 5003 instead of 5000.
        # padding with 0, each review has max_length now.
        words_index = sequence.pad_sequences(words_index, maxlen=self.max_length, padding='post',
                                                     truncating='post')
        return words_index

    def prepare_train_data(self):
        texts,labels = load_corpus()
        volcabulary, train_words = get_volcabulary_and_list_words(texts)

        self.set_volcabulary(volcabulary)

        del volcabulary,texts
        words_index = self.get_word_index(train_words, self.volcabulary, self.max_words, self.max_length)
    
        # del reviews_words, volcabulary
 
        index = np.arange(words_index.shape[0])
        train_index, valid_index = train_test_split(
            index, train_size=0.8, random_state=520)
        train_data = words_index[train_index]
        valid_data = words_index[valid_index]
        labels = np.asarray(labels)
        train_labels = labels[train_index]
        valid_labels = labels[valid_index]
        print(train_data.shape)
        print(valid_data.shape)
    
        pickle.dump((words_index, labels), open("output/zh_comments.pkl", 'wb'))
    
        return train_data, train_labels, valid_data, valid_labels
        
    def baseModel(self, nb_filter=250, filter_length=3, hidden_dims=125):
        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(self.max_words + self.index_from,self.embedding_dims,
                            input_length=self.max_length))
        model.add(Dropout(0.25))

        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:

        # filter_length is like filter size, subsample_length is like step in 2D CNN.
        model.add(Convolution1D(filters=nb_filter,
                                kernel_size=filter_length,
                                padding='valid',
                                activation='relu',
                                strides=1))
        # we use standard max pooling (halving the output of the previous layer):
        model.add(MaxPooling1D(pool_size=2))

        # We flatten the output of the conv layer,
        # so that we can add a vanilla dense layer:
        model.add(Flatten())

        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.25))
        model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop')

        return model

    def load_data(self,hdf5_path='output/zh_comments.pkl'):
        (words_index, labels) = pickle.load(open(hdf5_path, 'rb'))

        index = np.arange(words_index.shape[0])
        train_index, valid_index = train_test_split(
            index, train_size=0.8, random_state=520)
        train_data = words_index[train_index]
        valid_data = words_index[valid_index]
        labels = np.asarray(labels)
        train_labels = labels[train_index]
        valid_labels = labels[valid_index]
        return train_data, train_labels, valid_data, valid_labels

    def train_model(self, batch_size=32, nb_epoch=50,load_data = False,old_weight_path=''):
        print("start training model...")

        if load_data:
            train_data, train_labels, valid_data, valid_labels = self.load_data()
        else:
            train_data, train_labels, valid_data, valid_labels = self.prepare_train_data()

        model = self.baseModel()

        if old_weight_path != '':
            print("load last epoch model to continue train")
            model.load_weights(old_weight_path)

        model.fit(train_data, train_labels, batch_size=batch_size,
                  epochs=nb_epoch,
                  validation_data=(valid_data, valid_labels),
                  callbacks=[ModelCheckpoint("output/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True, save_weights_only=False, mode='min', period=2),
                             ProgbarLogger()])

        return model

    def predict_label(self,model,texts):
        words = get_list_words(texts)
        if not self.volcabulary:
            texts,labels = load_corpus()
            volcabulary, train_words = get_volcabulary_and_list_words(texts)
            self.set_volcabulary(volcabulary)
            del volcabulary,texts,labels

        words_index = self.get_word_index(words)
        #label = model.predict(words_index,batch_size=1)
        label = model.predict_classes(words_index,batch_size=1)
        return label