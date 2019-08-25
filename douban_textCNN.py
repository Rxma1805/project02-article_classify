import pandas as pd
data = pd.read_csv('douban_movie_comments.csv',encoding='gb18030')
data['comment'] = data['comment'].fillna('')
from collections import Counter
Counter(data['star'])
import numpy as np
from langconv import Converter
def tradition2simple(line):
    # 将繁体转换成简体
    line = Converter('zh-hans').convert(line)

    return line
sub_data = data
Counter(sub_data['star'])
import jieba

def is_CN_char(ch):
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

def cut(string):
    return list(jieba.cut(string))

def get_stopwords(filename = "stopWords.txt"):
    stopwords_dic = open(filename, encoding= 'utf-8')
    stopwords = stopwords_dic.readlines()
    stopwords = [w.strip() for w in stopwords]
    stopwords_dic.close()
    return stopwords

def convert2simple(word):
    return tradition2simple(word)
stopwords = get_stopwords()
def clean_sentence(sentence):
    stopwords = get_stopwords()
    sentence = ''.join(filter(is_CN_char,sentence))
    sentence = convert2simple(sentence)
    words = [w for w in cut(sentence) if  w not in stopwords]
    words = ' '.join(words)
    return words

sub_data['comment'] = sub_data['comment'].apply(clean_sentence)
def word_to_id(vocab):
    counts = Counter(vocab)
    vocab = sorted(counts, key=counts.get, reverse=True)
    word_to_id = { word : i for i, word in enumerate(vocab)}
    id_to_word = {i:word for i,word in enumerate(vocab)}
    return word_to_id, id_to_word
vocab = ' '.join(sub_data['comment']).split()
vocab.append('unknown')
word_to_id, id_to_word = word_to_id(vocab)
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format('sgns.baidubaike.bigram-char', binary=False, unicode_errors='ignore')

import numpy as np
word_vector = []
vec = np.zeros((300,))
count1=0
count2=0
try:
    for val in id_to_word.values():
        if val in wv :
            word_vector.append(wv[val])
            count1+=1
        else:
            print(val)
            count2+=1
            for v in val:
                if v in wv:
                    vec += wv[v]
            word_vector.append(vec)


except Exception as e:
    print(e)
print(count1,count2)

word_vector = np.array(word_vector)
print(len(id_to_word))
print(word_vector.shape)
def comment_to_id(word_to_id,comments):
    comment_to_id = []
    for comment in comments:
        comment_to_id.append([word_to_id[word] for word in comment.split()] )
    return comment_to_id

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Conv1D,GlobalMaxPooling1D, Concatenate, Dropout
from keras.layers.merge import concatenate
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

comment_to_id = comment_to_id(word_to_id,sub_data['comment'])
embedding_layer = Embedding(len(id_to_word),
        300,
        weights=[word_vector],
        input_length=200,
        trainable=False)

maxlen=200
pad_comments = pad_sequences(comment_to_id,maxlen=maxlen)
pad_comments = np.array(pad_comments)
from sklearn.model_selection import train_test_split
labels = (np.arange(1,6) == np.array(sub_data['star'])[:,None]).astype(np.float32)
ids = range(len(pad_comments))
x_train,x_test,y_train,y_test = train_test_split(ids,labels,test_size = 0.2, shuffle=True)

x_train = np.array(pad_comments[x_train])
x_test = np.array(pad_comments[x_test])
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape,x_test.shape)
sequence_1_input = Input(shape=(200,
                                ), dtype='float')
embedded_sequences_1 = embedding_layer(sequence_1_input)
convs=[]
for kernel_size in [3, 4, 5]:
    c = Conv1D(128, kernel_size, activation='relu')(embedded_sequences_1)
    c = Conv1D(64, kernel_size, activation='relu')(c)
    c = Conv1D(32, kernel_size, activation='relu')(c)
    c = GlobalMaxPooling1D()(c)
    convs.append(c)
x = Concatenate()(convs)
output = Dense(5, activation='sigmoid')(x)
model = Model(inputs=sequence_1_input, outputs=output)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train,
          batch_size=1000,
          epochs=10,
#           callbacks=[early_stopping],
          validation_data=(x_test, y_test)
          # metrics=['accuracy']
        )
