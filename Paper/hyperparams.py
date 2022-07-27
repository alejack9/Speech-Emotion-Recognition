from models.paper_model import PaperModelFactory

model_factories = [
  PaperModelFactory()
]

seconds = [8, 5, 4]

train_val_test_sizes = [((300, 100, 80))]
