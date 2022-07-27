import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True

def show_or_save(output_file):
    if output_file == "":
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        logging.debug(f"Plot saved in {output_file}")
        plt.close()

def kde_plot(serie, output_file=""):
    sns.kdeplot(serie)
    show_or_save(output_file)

def plot_labels_distribution(labels_distribution, one_hot_mapper, output_file=""):
    _, axes = plt.subplots(1, 3, figsize=(30, 10))
    titles = ["Complete DataSet's Labels distribution", "Train Set's Labels distribution", "Validation Set's Labels distribution"]
    ds_names = ['complete', 'train', 'val']
    ds_labels_distributions = [
        pd.Series(
            data=labels_distribution[name][1],
            index=[one_hot_mapper.get(v, v) for v in labels_distribution[name][0]]
        ).sort_index()
        for name in ds_names]
    for i, (title, ds_l_dist) in enumerate(zip(titles, ds_labels_distributions)):
        axes[i].set_title(title)
        axes[i].pie(ds_l_dist.values, labels=ds_l_dist.index, autopct='%1.1f%%')

    show_or_save(output_file)

def plot_audio_waves(train_ds, one_hot_mapper, rows=3, cols=3, output_file=""):
    n = rows * cols
    _, axes = plt.subplots(rows, cols, figsize=(12, 15))

    for i, (audio, label) in enumerate(train_ds.take(n)):
     r = i // cols
     c = i % cols
     ax = axes[r][c]
     ax.plot(audio.numpy())
     label = one_hot_mapper.get(np.array_str(label.numpy()), np.array_str(label.numpy()))
     ax.set_title(label)

    show_or_save(output_file)

def plot_speakers_pie(df, rows=2, cols=2, output_file=""):
    _, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, speaker in enumerate(df['speaker'].unique()):
        values = df[df['speaker'] == speaker]['label'].value_counts().sort_index()

        r = i // cols
        c = i % cols
        ax = axes[r][c]

        ax.set_title(speaker)
        ax.pie(values.values, labels=values.index,
               autopct=lambda p: '{:.0f}'.format(p * values.values.sum() / 100))

    show_or_save(output_file)
