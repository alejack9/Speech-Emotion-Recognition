import tensorflow as tf
import numpy as np
from consts import DATA_DIR, WORKING_DIR, SEED, EPOCHS, BATCH_SIZES, LOGGING_LEVEL
import os
import logging
import program
from os.path import join
from datetime import datetime

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(SEED)
np.random.seed(SEED)

logging.getLogger().setLevel(LOGGING_LEVEL)
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S')

program.run(DATA_DIR, join(WORKING_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")), EPOCHS, BATCH_SIZES)
