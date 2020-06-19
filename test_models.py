from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


def get_latest_checkpoint(checkpoints: Path):
    return max(checkpoints.iterdir(), key=lambda ckpt: ckpt.name.split('-')[1])


data = np.load('dataset/WISDM_ar_v1.1_preprocessed.npz')
x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']

label_encoder = tfds.features.ClassLabel(names_file='dataset/activity.labels.txt')

for model_type in Path('model').iterdir():
    latest_checkpoint = get_latest_checkpoint(model_type)
    model = load_model(str(latest_checkpoint))
    model.summary()
    y_pred = np.argmax(model.predict(x_test), axis=-1)

    conf_mat = confusion_matrix(y_test, y_pred)
    print(f' {model_type.name} '.center(26, '='))
    print(conf_mat)
    for i, row in enumerate(conf_mat):
        print(f'{label_encoder.int2str(i)}: {row[i] / sum(row) * 100:.2f}%')
    print('-' * 26)
    print(f'Overall: {np.sum(y_test == y_pred) / len(y_test) * 100:.2f}%')
    plot_confusion_matrix(conf_mat, colorbar=True, show_normed=True, class_names=label_encoder.names)
    print()

plt.show()
