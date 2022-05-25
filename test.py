import numpy as np
import tensorflow as tf
from tf2_utils import get_now_datetime, ImagePool, to_binary, load_npy_data_starGAN, save_midis, tf_shuffle_axis, concat_with_label


data_1 = np.array(['a','b','c'])
data_2 = np.array(['x','y','z'])

unchangeable_tensors = tf.constant([['a','b','c'],['x','y','z']])

print(unchangeable_tensors)
a = tf.reshape(unchangeable_tensors,[6])
print(a)
