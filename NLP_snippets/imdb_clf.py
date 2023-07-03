#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


# In[3]:


tf.__version__


# ### Fetch the data

# In[4]:


data, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)


# In[ ]:


print(data)


# ### Split the data into traingin and testing

# In[ ]:


train_data, test_data = data['train'], data['test']
print(train_data)


# In[ ]:


def parse_review(dataset):
    reviews = []
    labels = []

    for review, label in dataset:
        reviews.append(review.numpy().decode('utf8'))
        labels.append(label.numpy())
    return reviews, labels


# In[ ]:


train_review, train_label = parse_review(train_data)
test_review, test_label = parse_review(test_data)


# In[ ]:


print(len(train_review), len(train_label))
print(len(test_review), len(test_label))


# ### Tokenize words and padthem

# In[ ]:


numwords = 20000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=numwords, oov_token='<OOV>')
pad = tf.keras.preprocessing.sequence.pad_sequences

tokenizer.fit_on_texts(train_review)

train_seq = tokenizer.texts_to_sequences(train_review)
test_seq = tokenizer.texts_to_sequences(test_review)


# In[ ]:


mx_len = 0
for i in train_seq:
    mx_len = max(mx_len, len(i))


# In[ ]:


train_pad = pad(train_seq, padding='post', maxlen=120, truncating='post')
test_pad = pad(test_seq, padding='post', maxlen=120, truncating='post')

print(train_pad.shape)
print(test_pad.shape)


# In[ ]:


np.array(test_label).shape


# ### Define Model

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=numwords+1, output_dim=10, input_length=120),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

loss = tf.keras.losses.BinaryCrossentropy()
optim = tf.keras.optimizers.Adam(1e-4)

model.compile(
    loss = loss,
    optimizer=optim,
    metrics=['acc']
)


# In[ ]:


print(model.summary())


# ### Train Model

# In[ ]:


train_label_mdl = np.array(train_label,dtype=np.float32)
test_label_mdl = np.array(test_label,dtype=np.float32)
history = model.fit(
    train_pad,
    train_label_mdl,
    epochs=15,
    batch_size=250,
    validation_data=(test_pad, test_label_mdl)
)


# In[21]:


print(model.summary())


# ### Train Model

# In[22]:


train_label_mdl = np.array(train_label,dtype=np.float32)
test_label_mdl = np.array(test_label,dtype=np.float32)
history = model.fit(
    train_pad,
    train_label_mdl,
    epochs=15,
    batch_size=250,
    validation_data=(test_pad, test_label_mdl)
)


# In[ ]:





# In[ ]:




