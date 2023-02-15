from consts import SAVEE_DATA_DIR, ESD_DATA_DIR, SAVEE_WORKING_DIR, ESD_WORKING_DIR, SEED, EPOCHS, BATCH_SIZES, LOGGING_LEVEL
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

# conf type must be one of the possible type in the dict configuration of hyperparameters.py file
for conf_type in ["paper-model-data-pipe", "SAVEE-custom-model"]:
    program.run(SAVEE_DATA_DIR, SAVEE_WORKING_DIR, EPOCHS, BATCH_SIZES, "SAVEE", conf_type)

for conf_type in ["ESD-custom-model"]:
    program.run(ESD_DATA_DIR, ESD_WORKING_DIR, EPOCHS, BATCH_SIZES, "ESD", conf_type)