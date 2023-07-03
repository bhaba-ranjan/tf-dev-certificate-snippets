#!/usr/bin/env python
# coding: utf-8

# In[365]:


import numpy as np
import tensorflow as tf
import os
import random
from shutil import copyfile


# In[366]:


tf.__version__


# ### Seprate Images

# In[367]:


def create_folders(folder_names, root_path):
    for folder in folder_names:
        os.makedirs(os.path.join(root_path, folder))


# ### Sorting and checking for corrupted files (if training Directory contains a Mix of Binary classes)

# In[368]:


def is_corrupted(file_path):
    if os.path.getsize(file_path) == 0:
        print(f'{file_path} is corrupted')
        return True
    else:
        return False
    
def get_split_index(file_list, split_size):
    return int(len(file_list) * split_size)


def check_and_sort(files, root_path, cat_dir, dog_dir):
    
    copied_files = 0

    for file in files:
        
        # Your checking logic here to seprate files according to the criteria
        initials = file.split('.')[0]
        file_path = os.path.join(root_path, file)

        # copying files into two different dirs to use ImageGenerator
        if initials == 'cat' and is_corrupted(file_path) == False:            
            copyfile(file_path, os.path.join(cat_dir, file))
            copied_files += 1
        
        if initials == 'dog' and is_corrupted(file_path) == False:
            copyfile(file_path, os.path.join(dog_dir, file))
            copied_files += 1
    
    print(f'Total Files: {len(files)} \nTotal Copied Files: {copied_files}')


# In[369]:


def get_pre_trained_model(input_shape = None):
    
    # block_11_add for MobilenetV2
    # mixed7 for inceptionV3
    
    pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(
        input_shape= input_shape,
        include_top= False,
        weights='imagenet',        
    )
    

    # pre_trained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    #     input_shape= input_shape,
    #     include_top= False,
    #     weights='imagenet',        
    # )

    # pre_trained_model = tf.keras.applications.resnet50.ResNet50(
    #     input_shape= input_shape,
    #     include_top= False,
    #     weights='imagenet',        
    # )

    # pre_trained_model = tf.keras.applications.efficientnet.EfficientNetB0(
    #     input_shape= input_shape,
    #     include_top= False,
    #     weights='imagenet',        
    # )


    # set layers trainable to false
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # print(pre_trained_model.summary())

    return pre_trained_model


# In[370]:


def get_layer_output( pre_trained_model, layer_name):
    selected_layer = pre_trained_model.get_layer(layer_name)
    output = selected_layer.output
    return output
    


# In[371]:


# This is the path of the dataset
root_path_src = '../datasets/dogs-vs-cats/train/'

# This two paths need to exist inside of which the folders will be created

root_path_train = '../datasets/dogs-vs-cats-seprated/train/'
root_path_validation = '../datasets/dogs-vs-cats-seprated/validation/'
folder_names = ['cats', 'dogs']


# create folders: **the root_path should exist in the system
create_folders(folder_names= folder_names, root_path=root_path_train)
create_folders(folder_names= folder_names, root_path=root_path_validation)

# Get a list of file names in the direcotry
files = os.listdir(root_path_src)

# Random sample files if needed
files = random.sample(files, len(files))

# Find the index to split the file
# Convert it into training and Validation Split
index = get_split_index(files, .50)


# Copy file from src to destination directory
# It also checks if the files are corrupted
check_and_sort( files[:index] ,root_path_src, os.path.join(root_path_train, 'cats'), os.path.join(root_path_train, 'dogs'))
check_and_sort( files[index:] ,root_path_src, os.path.join(root_path_validation, 'cats'), os.path.join(root_path_validation, 'dogs'))


# In[372]:


# Image data generator with inbuilt augmentations
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,
                                                                  rotation_range=40,
                                                                  horizontal_flip= True,
                                                                  vertical_flip=False,
                                                                  zoom_range=0.0,
                                                                  width_shift_range=0.0,
                                                                  height_shift_range=0.0,                                                                  
                                                                  fill_mode='nearest')

# Test / Validation generator should not have augmentation
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)


# In[373]:


# To create classes from already segregated path
train_images = train_generator.flow_from_directory(root_path_train,
                                                   target_size= (224,224),
                                                   batch_size=12, 
                                                   shuffle=True,
                                                   class_mode='binary')

test_images = test_generator.flow_from_directory(root_path_validation,
                                                   target_size= (224,224),
                                                   batch_size=12,
                                                   class_mode='binary')


# In[374]:


pre_trained_model = get_pre_trained_model(input_shape=(224,224,3))

print(pre_trained_model.summary())


# #### Model Check Point

# In[403]:


check_point_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath= 'best_model_as_per_val_accuracy.h5',# path to save the model if the check point matches
    save_best_only= True,    
    monitor= 'val_accuracy', # check if the it is logged as val_acc or val_accuracy
    mode= 'auto' # options {min, max, auto}
)


# In[404]:


# block_11_add for MobilenetV2
# mixed7 for inceptionV3


output_layer = get_layer_output(pre_trained_model, layer_name='mixed7')

x = tf.keras.layers.Flatten()(output_layer)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs = pre_trained_model.input, outputs=x)

loss = tf.keras.losses.BinaryCrossentropy()
optimiser = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)

model.compile(loss=loss, optimizer= optimiser, 
              metrics=[
                  'accuracy', 
                #   tf.keras.metrics.Precision(), 
                #   tf.keras.metrics.Recall() 
                  ])


# In[405]:


history = model.fit(train_images, 
                    epochs= 1, 
                    validation_data=test_images,
                    callbacks=[check_point_cb])


# In[406]:


loaded_model = tf.keras.models.load_model('./best_model_as_per_val_accuracy.h5')


# In[407]:


loaded_model.summary()


# In[ ]:


model.predict()

