from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

WISDM_PATH = 'dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
PROCESSED_FILE = 'dataset/WISDM_ar_v1.1_preprocessed.npz'
FS = 20
FRAME_SIZE = FS * 4
HOP_SIZE = FS * 2

# Dirty data refinement
with open(WISDM_PATH) as f, StringIO() as g:
    g.write('user,activity,time,x,y,z\n')
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        tokens = line.split(';')
        for t in tokens:
            if not t:
                continue
            data = t.split(',')[:6]
            if len(data) != 6 or data[2] == '0' or not all(data):
                continue
            g.write(f'{",".join(data)}\n')

    g.seek(0)
    df = pd.read_csv(g).astype({'user': np.int32, 'x': np.float32, 'y': np.float32, 'z': np.float32})
    df_length = len(df)


def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, data['time'], data['x'], 'X-Axis')
    plot_axis(ax1, data['time'], data['y'], 'Y-Axis')
    plot_axis(ax2, data['time'], data['z'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()


# Plot some data
# for activity in df['activity'].value_counts().index:
#     data_for_plot = df[df['activity'] == activity][:Fs * 10]
#     plot_activity(activity, data_for_plot)

# Encode labels
label_encoder = tfds.features.ClassLabel(names=df['activity'].unique())
label_encoder.save_metadata('dataset', 'activity')
# df['label'] = [label_encoder.encode_example(activity) for activity in df['activity']]

# Standardize data
df[['x', 'y', 'z']] /= 20

# Frame preparation
frames, labels = [], []
start_index, end_index = 0, 1

while end_index < df_length:
    record_start = df.iloc[start_index]
    # Find the end of a record
    while end_index < df_length:
        record_end = df.iloc[end_index]
        if record_start['user'] != record_end['user'] or record_start['activity'] != record_end['activity']:
            break
        end_index += 1

    # Extract a record
    record = df.iloc[start_index:end_index]

    # Framing
    for i in range(0, len(record) - FRAME_SIZE, HOP_SIZE):
        frames.append(
            record[['x', 'y', 'z']].values[i:i + FRAME_SIZE]
        )
        labels.append(
            label_encoder.encode_example(record_start['activity'])
        )

    print(f'Progress: {end_index} / {df_length} ({(end_index / df_length) * 100:.2f} %)')
    start_index = end_index
    end_index += 1

frames, labels = np.array(frames), np.array(labels)
print(f'Number of all data samples: {len(frames)}')

# Find the activity that has the minimum number of data count
data_counts = [np.count_nonzero(labels == i) for i in range(label_encoder.num_classes)]
min_activity_count = min(data_counts)
min_activity = data_counts.index(min_activity_count)

print(f'Number of each activity: { {label_encoder.int2str(i): count for i, count in enumerate(data_counts)} }')
print(f'Minimum data label: {label_encoder.int2str(min_activity)}')

# Only select the minimum number of records from each activities
balanced_frames, balanced_labels = [], []
for i in range(label_encoder.num_classes):
    balanced_frames.append(
        frames[labels == i][:min_activity_count]
    )
    balanced_labels.append(
        labels[labels == i][:min_activity_count]
    )
x, y = np.concatenate(balanced_frames), np.concatenate(balanced_labels)
print(f'Number of balanced data samples: {len(x)}')

# Train / Test split
print(f'Splitting data into train (80%) / test (20%) groups...')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(f'x_train: {x_train.shape} / y_train: {y_train.shape}')
print(f'x_test: {x_test.shape} / y_test: {y_test.shape}')

# Save
print(f'Saving to {PROCESSED_FILE}...')
np.savez(PROCESSED_FILE, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
