#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[4]:


example_sentences = [
    'I love my dog',    
    'I love my cat',
    'I love to go on hiking'
]

validation = [
    'I really love my dog',
    ' This word in not present',
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')


# In[5]:


tokenizer.fit_on_texts(example_sentences)


# In[6]:


print(tokenizer.word_index)


# In[10]:


encoded_scentences = tokenizer.texts_to_sequences(example_sentences)
validation_encoding = tokenizer.texts_to_sequences(validation)

print(f'Encoded_scentences: {encoded_scentences}')
print(f'Validation_encoding: {validation_encoding}')


# In[12]:


padded_encoded = pad_sequences(encoded_scentences)
print(f'{padded_encoded}')


# In[13]:


padded_encoded = pad_sequences(encoded_scentences, padding='post')
print(f'{padded_encoded}')


# In[14]:


padded_encoded = pad_sequences(encoded_scentences, padding='post', maxlen = 5)
print(f'{padded_encoded}')


# In[15]:


padded_encoded = pad_sequences(encoded_scentences, padding='post', maxlen = 5, truncating='post')
print(f'{padded_encoded}')


# In[ ]:




