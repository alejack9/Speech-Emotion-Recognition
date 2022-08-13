import libs.data_operations as data_ops

from models.paper_model import PaperModelFactory
from models.custom_model import CustomModelFactory

from functools import reduce
from itertools import product

# hyperparams for custom models

# TODO different combinations of layer - n_filters - size
no_conv_layers = [4, 5, 6]
no_filters = [32, 64, 128, 256]
filter_sizes = [3, 9, 12]

# def getLayersConf():

pool_sizes = [2, 4]
no_fc_layers = [2, 3]
fc_neurons = [20, 60]

activations = ['relu']  # most common
dropouts = [0, 0.2, 0.5]

#  Adam optimizer
learning_rates = [0.001]
b1s = [0.9]
b2s = [0.999]


def _getCustomModel(hp):
    model = CustomModelFactory()
    model.setHyperparams(hp)
    return model

def getCustomModels():
    combinations = product(no_conv_layers, no_fc_layers, fc_neurons, no_filters, filter_sizes, pool_sizes,
                           activations, dropouts, learning_rates, b1s, b2s)
    return [_getCustomModel({
            'conv_layers': nc,
            'fc_layers': nfc,
            'fc_neurons': fc_n,
            'no_filters': nfil,
            'filter_size': fsz,
            'pool_size': psz,
            'activation': act,
            'dropout': d,
            'lr': lr,
            'b1': b1,
            'b2': b2,
            }) for (nc, nfc, fc_n, nfil, fsz, psz, act, d, lr, b1, b2) in combinations]



combinations = {
  'model_factories': [
    # PaperModelFactory(),
    *getCustomModels()
  ],
  'seconds' : [3, 4, 7],
  'patiences' : [80],
  'dropouts': [0, 0.2, 0.5],
  'train_val_test_percentages' : [(80, 10, 10)],
  'data_operations_factories' : [
    ('crop', lambda _: [
        data_ops.Crop(),
    ]),
    # ('fade05_crop', lambda total_audio_frames: [
    #   data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
    #   data_ops.Crop(),
    # ]),
    # ('fade10_crop', lambda total_audio_frames: [
    #   data_ops.Fade(total_audio_frames * 0.10, total_audio_frames * 0.10),
    #   data_ops.Crop(),
    # ]),
    # ('fade15_crop', lambda total_audio_frames: [
    #   data_ops.Fade(total_audio_frames * 0.15, total_audio_frames * 0.15),
    #   data_ops.Crop(),
    # ]),
    # ('crop_fade05', lambda total_audio_frames: [
    #   data_ops.Crop(),
    #   data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
    # ]),
    # ('crop_fade10', lambda total_audio_frames: [
    #   data_ops.Crop(),
    #   data_ops.Fade(total_audio_frames * 0.10, total_audio_frames * 0.10),
    # ]),
    # ('crop_fade15', lambda total_audio_frames: [
    #   data_ops.Crop(),
    #   data_ops.Fade(total_audio_frames * 0.15, total_audio_frames * 0.15),
    # ]),
    # ('crop_norm', lambda _: [
    #     data_ops.Crop(),
    #     data_ops.Normalize()
    # ]),
    # ('fade05_crop_norm', lambda total_audio_frames: [
    #   data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
    #   data_ops.Crop(),
    #   data_ops.Normalize()
    # ]),
    # ('fade10_crop_norm', lambda total_audio_frames: [
    #   data_ops.Fade(total_audio_frames * 0.10, total_audio_frames * 0.10),
    #   data_ops.Crop(),
    #   data_ops.Normalize()
    # ]),
    # ('fade15_crop_norm', lambda total_audio_frames: [
    #   data_ops.Fade(total_audio_frames * 0.15, total_audio_frames * 0.15),
    #   data_ops.Crop(),
    #   data_ops.Normalize()
    # ]),
    # ('crop_fade05_norm', lambda total_audio_frames: [
    #   data_ops.Crop(),
    #   data_ops.Fade(total_audio_frames * 0.05, total_audio_frames * 0.05),
    #   data_ops.Normalize()
    # ]),
    # ('crop_fade10_norm', lambda total_audio_frames: [
    #   data_ops.Crop(),
    #   data_ops.Fade(total_audio_frames * 0.10, total_audio_frames * 0.10),
    #   data_ops.Normalize()
    # ]),
    # ('crop_fade15_norm', lambda total_audio_frames: [
    #   data_ops.Crop(),
    #   data_ops.Fade(total_audio_frames * 0.15, total_audio_frames * 0.15),
    #   data_ops.Normalize()
    # ]),
  ]
}

total = reduce(lambda a, b: a * b, list(map(len, combinations.values())), 1)
