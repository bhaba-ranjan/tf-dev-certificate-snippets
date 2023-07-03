#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



file_path = './sonnets.txt'

tokenizer = Tokenizer() # no word limit for generation
lines = []

with open(file_path) as f:    
    lines =  f.read()
    lines = lines.lower().split('\n')


# In[ ]:





# In[ ]:





# In[ ]:


len(lines)


# ### Tokenize

# In[ ]:


tokenizer.fit_on_texts(lines)
sequenced_data = tokenizer.texts_to_sequences(lines)


# In[8]:


numwords = len(tokenizer.word_index)
print(f'number of words is: {numwords}')

sequenced_data  


# ### Generate dataset

# In[ ]:


preprocessed = []
for line in sequenced_data:
    for i  in range(1, len(line)):
        preprocessed.append(line[:i+1])


# ### extract the last label

# In[10]:


preprocessed


# In[11]:


max_length = 11
padded_train = pad_sequences(preprocessed, maxlen=max_length, truncating='post')


# In[12]:


print(f'features have shape: {padded_train.shape}')


# In[13]:


padded_features = padded_train[:,:-1]
label = np.array(padded_features[:,-1], dtype=np.float32)


# In[14]:


embedding_dim = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=numwords+1, output_dim=embedding_dim), 
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
    tf.keras.layers.LSTM(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(numwords+1, activation='softmax')
])

loss = tf.keras.losses.SparseCategoricalCrossentropy()
optim = tf.keras.optimizers.Adam(1e-2)

model.compile(
    loss=loss,
    optimizer=optim,
    metrics=['acc']
)


# In[15]:


model.fit(
    padded_features,
    label,
    epochs=30,
    batch_size=100
)


# In[ ]:




