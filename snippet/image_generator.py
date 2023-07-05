import tensorflow as tf

train_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,
                                                                horizontal_flip=True,
                                                                vertical_flip=False,
                                                                width_shift_range=0.3,
                                                                height_shift_range=None,
                                                                zoom_range=None,
                                                                fill_mode='nearest')

validation_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)



# Example read from directory
####  @param: CLASS MODE should be specified here

# train_images = train_generator.flow_from_directory(root_path_train,
#                                                    target_size= (224,224),
#                                                    batch_size=12,
#                                                    shuffle=True,
#                                                    class_mode='binary')
#
# test_images = validation_img_gen.flow_from_directory(root_path_validation,
#                                                  target_size= (224,224),
#                                                  batch_size=12,
#                                                  class_mode='binary')


# Example read fom in-memory data and labels

# # train_images = train_img_gen.flow(x=train_x,
# #                                   y=train_y,
# #                                   batch_size=64)
# #
# # validation_images = validation_img_gen.flow(x = val_x,
# #                                             y = val_y,
# #                                            batch_size=64)


