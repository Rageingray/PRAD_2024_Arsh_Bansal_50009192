import librosa
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output
import glob
import imageio
import time
import IPython.display as ipd
AUTOTUNE = tf.data.experimental.AUTOTUNE
