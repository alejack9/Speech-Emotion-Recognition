from consts import DATA_DIR, WORKING_DIR, SEED, EPOCHS, BATCH_SIZES, LOGGING_LEVEL
import tensorflow as tf
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger().setLevel(LOGGING_LEVEL)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S')

import numpy as np
np.random.seed(SEED)

import program
program.run(DATA_DIR, WORKING_DIR, EPOCHS, BATCH_SIZES)



