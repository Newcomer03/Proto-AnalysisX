import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras
import re

url = 'customer complaints data.xlsx'

df = pd.read_excel(url , header =0 )

df.head

df.dropna()

df.shape

def cleanText(text):
  text = re.sub(r'@[A-Za-z0-9]+', ' ', text)     # removing @mentions
  text = re.sub(r'#', ' ', text )                # removing '#' sign
  text = re.sub(r'RT[\s]+', ' ', text)           # removing RT
  text = re.sub(r'https?\/\/\S+', ' ', text)     # removing the hyper link
   

  return text
df['TWEET'] =  df['TWEET'].apply(str)

df['TWEET'] =  df['TWEET'].apply(cleanText)

df.head


A= np.array(df['TWEET'])
B= np.array(df['ENCODE'])
print(A.shape)
print(B.shape)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_vocab = 2000000
tokenizer = Tokenizer(num_words= max_vocab)
tokenizer.fit_on_texts(A)


wordix = tokenizer.word_index
V=len(wordix)


train_seq = tokenizer.texts_to_sequences(A)



pad_train = pad_sequences(train_seq)
T =  pad_train.shape[1]


from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, GlobalMaxPooling1D,Embedding

D=20
M=15

i = Input(shape = (T,))
x = Embedding(V+1, D)(i)    
x = LSTM(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(32, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i,x)

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

r = model.fit(pad_train,B, epochs = 2)

def predict_statement(text):
  text_seq = tokenizer.texts_to_sequences(text)
  text_pad = pad_sequences(text_seq, maxlen=T)


  predicted_sentiment = model.predict(text_pad).round()
  val = np.argmax(predicted_sentiment)
  if val==0:
    return "Negative Statement"
  else:
    return "Positive Statement"
  