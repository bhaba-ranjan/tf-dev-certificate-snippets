import tensorflow as tf
import numpy as np

check_point_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'best_model_as_per_val_accuracy.h5',  # path to save the model if the check point matches
    save_best_only=True,
    monitor='val_accuracy',  # check if it is logged as val_acc or val_accuracy
    mode='auto'  # options {min, max, auto}
)

##
LR = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20), verbose=1)
lrs = 1e-5 * 10**(np.arange(num_epocs_here) / 20)

# plotting example
# import matplotlib.pyplot as plt
# lrs = 1e-5 * 10**(np.arange(15) / 20)

# plt.semilogx(lrs, history.history["loss"])
# plt.show()


## Pass the OBJECT of this class for using early stopping
class EarlyStoppingMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        super(EarlyStoppingMonitor, self).__init__()
        self.current_best = 0
        self.monitor = 'acc'

    def on_epoch_end(self, epoch, logs=None):
        # get parameter like below
        # current_train_precision = logs.get('precision')
        # self.model.stop_training = True

        if self.current_best < logs.get(self.monitor):
            self.current_best = logs.get(self.monitor)
            self.model.save(f'best_model_as_{self.monitor}_{self.current_best:.3f}.h5')
        # print('\n\n******* Stopping on Defined Threshold *******')

    def on_train_end(self, logs=None):
        if self.model.stop_training:
            print("\n\n\n****** Early Stopping *******")

# Example to use callbacks in the callbacks parameter
# @param: callbacks will take the callback object

# # history = model.fit(train_images,
# #                     epochs= 1,
# #                     validation_data=test_images,
# #                     callbacks=[check_point_cb])
