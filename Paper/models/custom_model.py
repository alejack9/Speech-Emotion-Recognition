from models.model_factory import ModelFactory
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import logging

class CustomModelFactory(ModelFactory):

    def get_model_name(self, _={}):
        return f"Model_convl{self.hp['conv_layers']}_fcl{self.hp['fc_layers']}_fcsize{self.hp['fc_neurons']}_filters{self.hp['no_filters']}_fsize{self.hp['filter_size']}_poolsize{self.hp['pool_size']}_act_{self.hp['activation']}_drop{self.hp['dropout']}_lr{str(self.hp['lr']).split('.')[1]}_b1{str(self.hp['b1']).split('.')[1]}_b2{str(self.hp['b2']).split('.')[1]}___"

    def setHyperparams(self, hp):
        self.hp = hp
        

    def get_model(self, args={'input_shape': (1, 1), "print_summary": False, "classes": 7}):
        '''input_shape = (sample_rate * seconds, 1)'''

#         {
#             'conv_layers': nc,
#             'fc_layers': nfc,

#             'fc_neurons': fc_n,

#             'no_filters': nfil,
#             'filter_size': fsz,
#             'pool_size': psz,

#             'activation': act,
#             'dropout': d,

#             'lr': lr,
#             'b1': b1,
#             'b2': b2,
#             }

        model = Sequential()

        # there is always the first conv layer
        model.add(Conv1D(
            self.hp['no_filters'], self.hp['filter_size'], activation=self.hp['activation'], input_shape=args['input_shape']))
        model.add(BatchNormalization())

        # the first is excluded

        for _ in range(self.hp['conv_layers'] - 1):
            model.add(MaxPooling1D(pool_size=self.hp['pool_size']))
            model.add(Conv1D(
                self.hp['no_filters'], self.hp['filter_size'], activation=self.hp['activation'],))
            model.add(BatchNormalization())

        model.add(GlobalMaxPooling1D())

        #  fc layer
        for _ in range(self.hp['fc_layers']):
            model.add(Dense(self.hp['fc_neurons'],
                      activation=self.hp['activation']))
            model.add(Dropout(self.hp['dropout']))

        #  last dense layer
        model.add(Dense(args["classes"], activation='softmax'))

        if args.get("print_summary", False):
            model.summary()

        opt = Adam(learning_rate=self.hp['lr'],
                   beta_1=self.hp['b1'], beta_2=self.hp['b2'])

        model.compile(
            optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model
