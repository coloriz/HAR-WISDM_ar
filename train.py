from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import InputLayer, Conv1D, BatchNormalization, ReLU, Dense, Flatten, Softmax, LSTM
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

parser = ArgumentParser(description='Train a model')
parser.add_argument('model_type', choices=['ModelA', 'ModelB'], help='model type to train')
args = parser.parse_args()

model_type = args.model_type

data = np.load('dataset/WISDM_ar_v1.1_preprocessed.npz')
x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']

if model_type == 'ModelA':
    model_path = Path('model/ModelA')
    log_dir = 'log/ModelA'
    model = Sequential([
        InputLayer(input_shape=(80, 3)),
        Conv1D(16, 3),
        BatchNormalization(),
        ReLU(),
        Conv1D(32, 3),
        BatchNormalization(),
        ReLU(),
        Flatten(),
        Dense(6),
        Softmax()
    ], 'WISDM-ModelA')
else:
    model_path = Path('model/ModelB')
    log_dir = 'log/ModelB'
    model = Sequential([
        InputLayer(input_shape=(80, 3)),
        Conv1D(16, 3),
        BatchNormalization(),
        ReLU(),
        Conv1D(32, 3),
        BatchNormalization(),
        ReLU(),
        LSTM(128),
        Dense(6),
        Softmax()
    ], 'WISDM-ModelB')
model_path.mkdir(parents=True, exist_ok=True)
model.compile(optimizer=Adam(0.0005),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
callbacks = [
    TensorBoard(log_dir=log_dir),
    EarlyStopping(monitor='val_accuracy', patience=20),
    ModelCheckpoint(str(model_path / 'model-{epoch:03d}-{val_accuracy:.4f}'), 'val_accuracy', save_best_only=True)
]
model.fit(x_train, y_train, epochs=100, callbacks=callbacks, validation_data=(x_test, y_test))
