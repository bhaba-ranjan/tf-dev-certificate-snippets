import tensorflow as tf

# Tokenizer and padding sequence basics
# Do not pass Numworkes or OOV for language generation tasks
numwords = 20000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=numwords, oov_token='<OOV>')
pad = tf.keras.preprocessing.sequence.pad_sequences


# Finding maximum length tokenized by WORDS
# This can be used for padding words
def find_max_len(train_seq):
    mx_len = 0
    for i in train_seq:
        mx_len = max(mx_len, len(i))
    print(mx_len)
    return mx_len



## Tokenizer example
# tokenizer.fit_on_texts(train_x)
#
# train_seq = tokenizer.texts_to_sequences(train_x)
# test_seq = tokenizer.texts_to_sequences(val_x)


## Padding example
## For Language GENERATION tasks the padding should be 'pre'
# train_seq_pad = pad(train_seq, padding='post', truncating='post', maxlen=mx_len)
# test_seq_pad = pad(test_seq, padding='post', truncating='post', maxlen=mx_len)
