import matplotlib.pyplot as plt

def speaker_pie_plot(df, rows=2, cols=2):
    _, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, speaker in enumerate(df['speaker'].unique()):
        values = df[df['speaker'] == speaker]['label'].value_counts()

        r = i // cols
        c = i % cols
        ax = axes[r][c]

        ax.set_title(speaker)
        ax.pie(values.values, labels=values.index,
               autopct=lambda p: '{:.0f}'.format(p * values.values.sum() / 100))

    plt.show()
