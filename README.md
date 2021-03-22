# Knowledge-Distillation-aided-End-to-End-Learning-for-Linear-Precoding-in-Multiuser-MIMO-Downlink-System

With Knowledge distillation

Run Limited_feedback_MU_Expanded_Final_8_2_4_scrating_hint_same_network.py

Without Knowledge distillation

Run Limited_feedback_MU_Expanded_Final_8_2_4_scrating_no_hint_same_network.py

Environment

This code is tested in Tensorflow 2.5 with RTX3090, but Tensorflow version 1 can also be used with small modifications.

Change

import tensorflow.compat.v1 as tf, tf.disable_v2_behavior()

TO

import tenforflow as tf


Dataset can be downloaded as following link:

https://drive.google.com/drive/folders/1eEAgOPUgmwnQb0GdvhZqBbhBuGWKWTHn?usp=sharing
