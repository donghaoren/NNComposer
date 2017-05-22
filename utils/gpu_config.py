import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def tfSessionAllowGrowth():
    def get_session():
        gpu_options = tf.GPUOptions(allow_growth = True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    KTF.set_session(get_session())
