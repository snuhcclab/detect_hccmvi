# SNU HCC LAB Junseo Kang
# Receive Model and Hyperparams, Compile and Fit
# Not used on Shell
import os.path

from keras.models import Model, save_model
from keras.metrics import BinaryAccuracy
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping
from numpy import concatenate
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from os import path


def trainer(experiment, result_dir):
    """
    :param experiment:
    :param result_dir: result_dir
    :return:
    """
    callbacks = []
    if experiment.tensorboard:
        try:
            callbacks.append(
                TensorBoard(log_dir=os.path.join(result_dir, 'logs'))
            )
        except KeyError:
            callbacks.append(
                TensorBoard(log_dir='.')
            )

    if experiment.early_stopping:
        callbacks.append(
            EarlyStopping(patience=experiment.early_stopping)
        )

    experiment.model.compile(
        optimizer=getattr(optimizers, experiment.optimizer)(learning_rate=experiment.learning_rate,
                                                            clipnorm=1.0),
        loss=experiment.loss,
        metrics=[BinaryAccuracy(), 'accuracy'])
    print(next(iter(experiment.train_data)))

    history = experiment.model.fit(experiment.train_data,
                                   validation_data=experiment.val_data,
                                   epochs=experiment.epoch,
                                   callbacks=callbacks)

    save_model(experiment.model, os.path.join(result_dir, 'model'))

    return experiment.model, history


def plotter(experiment, hist, result_dir):
    """
    :param experiment:
    :param hist:
    :param result_dir:
    :return:
    """
    # Estimate Model
    #
    evaluation = experiment.model.evaluate(experiment.test_data)

    # test_x = concatenate([x['image'] if isinstance(x, dict) else x for x, y in experiment.test_data], axis=0)
    test_y = concatenate([y for x, y in experiment.test_data], axis=0)
    expect_y = experiment.model.predict(experiment.test_data).tolist()

    # tn, fp, fn, tp = confusion_matrix(test_y, expect_y).ravel()

    #
    # Right Estimation Report
    #

    #
    # Plot Training History
    #
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    plt.savefig(path.join(result_dir, 'train_loss.png'))
    plt.clf()

    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    plt.savefig(path.join(result_dir, 'train_acc.png'))
    plt.clf()

    return 0

