import numpy as np

SEED = 0

DATA_DIR = "./data/ESD_data"
WORKING_DIR = './stuff'

EPOCHS = 400

BATCH_SIZES = np.logspace(start=5, stop=0, num=6, base=2, dtype=np.int32)

# LOGGING_LEVEL = "DEBUG"
LOGGING_LEVEL = "INFO"
