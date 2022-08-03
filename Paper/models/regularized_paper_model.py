from models.model_factory import ModelFactory
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
import logging

#  kernel_regularizer=regularizers.l2(l=0.01)
#  dropout

class PaperModelFactory(ModelFactory):
  def get_model_name(self, _={}):
     return f"PaperModelRegularized_d{self.dropout}_l2{str(self.l2_factor).spli('.')[1]}"

  def set_conf(self, dropout = 0.2, l2 = 0.01):
    self.dropout = dropout
    self.l2_factor = l2;

  def get_model(self, args={'input_shape': (1, 1), "print_summary": False}):
    '''input_shape = (sample_rate * seconds, 1)'''
    
    filters = [32, 64, 128, 256, 512, 1024, 1024]
    sizes = [21, 19, 17, 15, 13, 11, 9]
    activation = 'relu'
    pool_size = 2

    model = Sequential()

    # first layer
    # input shape (None, n) = variable-length sequences of n-dimensional vectors
    model.add(Conv1D(filters[0], sizes[0], activation=activation, input_shape=args['input_shape'], kernel_regularizer=L2(l2=self.l2_factor)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size))

    # middle layers
    for (filter_size, kernel_size) in list(zip(filters, sizes))[1:-1]:
        model.add(Conv1D(filter_size, kernel_size, activation=activation, kernel_regularizer=L2(l2=self.l2_factor)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=pool_size))

    # last layer
    model.add(Conv1D(filters[-1], sizes[-1], activation=activation, kernel_regularizer=L2(l2=self.l2_factor)))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())

    model.add(Dropout(self.dropout))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(self.dropout))
    model.add(Dense(7, activation='softmax'))

    if args.get("print_summary", False):
      model.summary()

    opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

    return model