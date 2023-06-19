import time
from typing import List, Any, Callable
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction, SparseCategoricalCrossentropy
from tensorflow.python.keras import regularizers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers.experimental import SGD
from first_order_inf import FirstOrderInfluence
from signals import InfluenceErrorSignals


def build_lr_model(
        classes: int,
        input_shape: int,
        regularization=False
):
    reg_strength = 0.01 if regularization else 0
    initializer = tf.keras.initializers.GlorotNormal(seed=42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=classes,
                input_shape=[input_shape],
                kernel_initializer=initializer,
                kernel_regularizer=regularizers.L2(reg_strength),
                bias_regularizer=regularizers.L2(reg_strength),
                activation="softmax",
            ),
        ]
    )
    return model

def split_stratify_data(X, y):
    # Split to 80% train 20% test set and stratify
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    return X_train, X_test, y_train, y_test


def convert_to_tf_dataset(X, y, num_classes):
    # Tensorflow has a bug for reduction.NONE. it keeps reducing the loss using avg aggregation
    # bypassing the parameter NONE. To fix it, we use one hot encoding to the labels. This change does not affect anything else
    y_one_hot_enc = preprocess_y(y, num_classes)
    return tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(X, dtype=tf.float32), y_one_hot_enc)
    )

def preprocess_y(y, num_classes):
    return tf.one_hot(tf.convert_to_tensor(y.astype(int)), depth=num_classes)

def read_data():
    df = pd.read_csv('digits.csv')
    if 'has_error' in df.columns:
        df = df.drop(columns=['has_error'])
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y


if __name__ == '__main__':

    X, y = read_data() # here you will replace read data with your synthetic data. Ensure that every time they are the same
    num_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = split_stratify_data(X, y)

    # In the experiments will calculate the influence of every data point of the train set to the test set
    train_set = convert_to_tf_dataset(X_train, y_train, num_classes)
    test_set = convert_to_tf_dataset(X_test, y_test, num_classes)

    model = build_lr_model(classes=num_classes, input_shape=X.shape[1], regularization=True)

    epochs = 20
    batch_size = 64
    learning_rate = 1e-2
    loss_fn_reduced = CategoricalCrossentropy()

    optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn_reduced, metrics=['accuracy'])

    # model.fit(
    #     train_set.batch(batch_size),
    #     validation_data=test_set.batch(batch_size),
    #     epochs=epochs
    # )

    loss_fn_unreduced = CategoricalCrossentropy(reduction=Reduction.NONE) # check tf documentation for the reduction

    test_losses = loss_fn_unreduced(preprocess_y(y_test, num_classes), model(tf.convert_to_tensor(X_test)))

    print(test_losses)

    ########################################################
    # Calculate the influence in a subset using FOIF
    ########################################################

    model = build_lr_model(classes=num_classes, input_shape=X.shape[1], regularization=True)
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn_reduced, metrics=['accuracy'])

    # note that foif will train the model internally

    foif = FirstOrderInfluence(
        model_obj=model,
        batch_size=batch_size,
        learning_rate=1e-2,
        epochs=epochs,
        unreduced_loss_fn=loss_fn_unreduced
    )
    print("First Order Influence: Building Influence Models")

    foif.build_influence_models(train_data=train_set, val_data=test_set)

    # we will take a subsample just to show the usage of FOIF fast
    train_samples_to_keep = 10
    test_samples_to_keep = 5
    subsample_train_idx = X_train.index[:train_samples_to_keep]
    subsample_test_idx = X_test.index[:test_samples_to_keep]

    subsample_train = convert_to_tf_dataset(
        X_train.loc[subsample_train_idx], y_train.loc[subsample_train_idx], num_classes
    )
    subsample_test = convert_to_tf_dataset(
        X_test.loc[subsample_test_idx], y_test.loc[subsample_test_idx], num_classes
    )

    # you need to give the data in batch dataset format, The batch size must be the same as the samples

    self_inf_matrix = foif.compute_self_influence(
        train_points=subsample_train.batch(len(subsample_train)),
    )

    self_inf_matrix.index = subsample_train_idx
    self_inf_matrix.columns = subsample_train_idx # since it is the self influence, the columns are the same as train indices

    train_to_test_inf_matrix = foif.compute_train_to_test_influence(
        train_points=subsample_train.batch(len(subsample_train)),
        test_points=subsample_test.batch(len(subsample_test)),
    )

    train_to_test_inf_matrix.index = subsample_train_idx
    train_to_test_inf_matrix.columns = subsample_test_idx

    print(self_inf_matrix.shape)
    print(self_inf_matrix)

    print(train_to_test_inf_matrix.shape)
    print(train_to_test_inf_matrix)

    # Computing influence signals based on influence values

    ies = InfluenceErrorSignals()

    si_sig = ies.compute_signals_per_sample(
        inf_mat=self_inf_matrix,
        train_samples_to_examine=subsample_train_idx,
        y_true=y,
        signals=['SI'],
        njobs=10
    )

    print(si_sig)

    rest_sigs = ies.compute_signals_per_sample(
        inf_mat=train_to_test_inf_matrix,
        train_samples_to_examine=subsample_train_idx,
        y_true=y,
        signals=ies.signals_names() - {'SI'},
        njobs=10
    )

    print(rest_sigs)