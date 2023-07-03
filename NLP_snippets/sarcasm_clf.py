#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import tensorflow as tf
import numpy as np


# In[2]:


file = open('../datasets/news_sarcasm_data_set/Sarcasm_Headlines_Dataset.json')

# print(file.__next__())


# ### Read Files

# In[4]:


headLine = []
label = []
lengths = []
with open('../datasets/news_sarcasm_data_set/Sarcasm_Headlines_Dataset.json') as file:
    for jsonObject in file:
        parsedObject = json.loads(jsonObject)
        headLine.append(parsedObject['headline'])
        lengths.append(len(headLine[-1].split(' ')))
        label.append(parsedObject['is_sarcastic'])

print(f'headlines: {len(headLine)}')
print(f'labels: {len(label)}')
print(f'maximum length of string by WORDS: {max(lengths)}')


# ### TF tokenize and padding

# In[5]:


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2000, oov_token='<OOV>')
tokenizer.fit_on_texts(headLine)
encodeHeadlines = tokenizer.texts_to_sequences(headLine)

encodeHeadlines = tf.keras.preprocessing.sequence.pad_sequences(encodeHeadlines, padding='pre', truncating='post', maxlen=40)
label = np.array(label, dtype=np.float32)

encodeHeadlines = np.array(encodeHeadlines, dtype=np.float64)

print(f'Shape of encoded sequences: {encodeHeadlines.shape}')
print(f'Shape of labels: {label.shape}')


# In[ ]:


loss = tf.keras.losses.BinaryCrossentropy()
optim =tf.keras.optimizers.Adam(learning_rate=0.001)

# model = tf.keras.Sequential([
#     tf.keras.layers.RepeatVector(1),
#     tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim= 2000, # Num words in tokenizer
        output_dim=20,
        input_length= encodeHeadlines.shape[1]
    ),
    tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.RepeatVector(1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, input_length=20)),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=loss,
    optimizer=optim,
    metrics=['acc']
)


# ### Convert both the inputs to NUMPY ARRAY

# In[ ]:


print(model.summary())


# In[ ]:


history = model.fit(x= encodeHeadlines,
                    y= label,
                    epochs=20,
                    validation_split=.3)

