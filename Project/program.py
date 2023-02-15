from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from libs.utils import create_folder
from models.callbacks import get_callbacks
from os.path import join
from datetime import datetime
from itertools import product
# from hyperparams import total as total_pars, combinations as hyperparams

from hyperparams import configurations

# from hyperparams import paper_model_total, custom_model_total
# from hyperparams import paper_model_combinations, custom_model_combinations

import libs.data_loader as data_loader
import libs.model_runner as model_runner
import libs.data_visualization as visual

import logging
import re
import os

data_utilities = {
    "ESD": (
        lambda file_path: re.sub('.*[/]+[a-z]\_|\_[0-9]+.wav', '', file_path), # get label
        lambda file_path: re.sub('.*[/]+|\_|[a-z]+_[0-9]+.wav', '', file_path) # get speaker name
    ),
    "SAVEE": (
        lambda file_path: re.sub('.+\_|[0-9]+.wav', '', file_path), # get label
        lambda file_path: re.sub('.*[/]+|\_|[a-z]+[0-9]+.wav', '', file_path) #get speaker name
    )
}

def data_analysis(plots_dir, df):
    visual.plot_speakers_pie(
        df, output_file=join(plots_dir, "speakers_pie.png"))
    visual.kde_plot(df['length'], output_file=join(
        plots_dir, "lengths_kde.png"))


def run(data_dir, working_dir, epochs, batch_sizes, ds_name, conf_type):
    ''' ds_name -> ESD or SAVEE \r\n
        conf_type -> paper-model-data-pipe,custom-model, custom-model-normalized-data
    '''

    plots_dir = create_folder(join(working_dir, "plots"))
    completed_models_dir = create_folder(join(working_dir, "done"))

    get_label, get_speaker_name = data_utilities[ds_name] # get SAVEE or ESD utility function

    logging.info("Loading data...")

    df, one_hot_mapper, max_sample_rate = data_loader.get_dataset_information(
        data_dir, get_label, get_speaker_name)
    logging.info("Loading data... done")
    
    n_classes = len(df['label'].unique())
    logging.info(f"{n_classes} classes.")

    data_analysis(plots_dir, df)

    # configurations = {
    #   "paper-model-data-pipe": (paper_model_combinations, paper_model_total),
    #   "SAVEE-custom-model": (SAVEE_custom_model_combinations, SAVEE_custom_model_total),
    #   "ESD-custom-model": (ESD_custom_model_combinations, ESD_custom_model_total),
    #   "ESD-paper-model": (ESD_paper_model_combinations, 1)
# }
    hyperparams, total_pars = configurations[conf_type]

    for i, (model_factory, train_val_tests_percentage, audio_seconds, (data_ops_name, data_ops_factory), patience, dropout) in enumerate(product(hyperparams['model_factories'], hyperparams['train_val_test_percentages'], hyperparams['seconds'], hyperparams['data_operations_factories'], hyperparams['patiences'], hyperparams['dropouts'])):
        computed = False
    
        for j, batch_size in enumerate(batch_sizes):
            logging.info(f"------------- Trying batch size {j + 1} / {len(batch_sizes)} ({round((j + 1) / float(len(batch_sizes)) * 100, 2)}%) ------------")
            logging.info(f'Model: {model_factory.get_model_name()} , seconds: {audio_seconds} , batch_size: {batch_size} , early_stopping_patience: {patience} data_ops: {data_ops_name}')

            train_ds, val_ds, test_ds, additional = data_loader.load_datasets(df, max_sample_rate, audio_seconds,
                                                                                data_ops_factory, train_val_tests_percentage)

            model = model_factory.get_model(args={"input_shape": (max_sample_rate * audio_seconds, 1), 'dropout': dropout, 'print_summary': False, 'classes': n_classes})
            model_name = f"m{model_factory.get_model_name()}_s{audio_seconds}_b{batch_size}_d{dropout}_p{patience}_o_{data_ops_name}_sz{str(train_val_tests_percentage).replace(' ', '')[1:-1]}"

            done_model_path = join(completed_models_dir, model_name)

            if os.path.exists(done_model_path):
                logging.info("Model already computed.")
                computed = True
                break

            current_plots_dir = create_folder(join(plots_dir, model_name))
            checkpoints_dir = create_folder(
                join(working_dir, "checkpoints", model_name))
            logs_dir = create_folder(
                join(working_dir, "logs", model_name, datetime.now().strftime("%Y%m%d-%H%M%S")))
            last_epoch_file_path = join(create_folder(
                join(working_dir, "last_epochs")), f"{model_name}.txt")

            visual.plot_labels_distribution(additional['labels_distribution'], one_hot_mapper, output_file=join(
                current_plots_dir, "labels_distribution.png"))
            visual.plot_audio_waves(train_ds, one_hot_mapper, output_file=join(
                current_plots_dir, "audio_waves.png"))

            train_ds = train_ds.batch(batch_size)
            val_ds = val_ds.batch(batch_size)

            init_epoch = 0

            if os.path.exists(last_epoch_file_path):
                model.load_weights(checkpoints_dir)
                with open(last_epoch_file_path, "r") as f:
                    init_epoch = int(f.readline())

            logdir = join(logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
            cbs = get_callbacks(checkpoints_dir, logdir, last_epoch_file_path, patience)

            try:
                model_runner.run(model, train_ds, val_ds, cbs=cbs, epochs=epochs, init_epoch=init_epoch)
                computed = True
            except ResourceExhaustedError:
                logging.error("Not enough GPU memory.")

            if computed == True:
                open(done_model_path, "w+").close()

        if not computed:
            logging.critical(
                "No batch size is suitable for this hardware. Trying next hyper-parameters configuration.")

        logging.info(
            f"-------------------- {str(i + 1).rjust(len(str(total_pars)), ' ')} / {total_pars} ({round((i + 1) / float(total_pars) * 100, 2)}%) --------------------")
