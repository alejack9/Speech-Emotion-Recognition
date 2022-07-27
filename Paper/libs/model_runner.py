import logging

def run(model, train_ds, val_ds, cbs=[], epochs=400, init_epoch=0):
  model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=cbs, initial_epoch=init_epoch)