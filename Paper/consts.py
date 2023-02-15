import numpy as np

SEED = 0

SAVEE_DATA_DIR = "./data/SAVEE"
ESD_DATA_DIR = "./data/ESD"

SAVEE_WORKING_DIR = "./SAVEE_project_data/"
ESD_WORKING_DIR = "./ESD_project_data/"

EPOCHS = 400

BATCH_SIZES = np.logspace(start=5, stop=0, num=6, base=2, dtype=np.int32)

# LOGGING_LEVEL = "DEBUG"
LOGGING_LEVEL = "INFO"
