#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import tensorflow as tf
import tf2onnx


tf.random.set_seed(43)

def main():
    data = pd.read_csv("training.csv.gz")
    x = np.asarray(data.loc[:, data.columns != 'Label'], dtype=np.float32)
    le = preprocessing.LabelEncoder()
    le.fit(data.Label)
    y = le.transform(data.Label).astype(np.float32)
    labels = np.array(le.classes_, dtype=str)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    INPUT_DIM  = x_train.shape[1]
    HIDDEN_DIM = 100
    OUTPUT_DIM = len(labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(HIDDEN_DIM, activation="linear"),
        tf.keras.layers.Dense(OUTPUT_DIM, activation="softmax", name="output"),
    ])

    learning_rate = 1e-4
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, tf.keras.utils.to_categorical(y_train, num_classes=len(labels)),
        epochs=3, batch_size=10,
        validation_data=(x_test, tf.keras.utils.to_categorical(y_test, num_classes=len(labels))))

    spec = (tf.TensorSpec((None, INPUT_DIM), tf.float32, name="input"),)
    output_path = "particles.onnx"

    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

if __name__ == "__main__":
    main()
