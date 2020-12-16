#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhenghao Lu'


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import cm


# plot sample 2d and 3d images
def plot_sample(band_1, band_2, band_3, labels):
    # pick the random images from positive set and negative set
    index = np.arange(len(labels))
    ice_index = index[labels == 1]
    ship_index = index[labels == 0]
    ice_sample = np.random.choice(ice_index, size=1, replace=False)[0]
    ship_sample = np.random.choice(ship_index, size=1, replace=False)[0]
    X = np.arange(75)
    Y = np.arange(75)
    x, y = np.meshgrid(X, Y)
    # show 3d surface plots
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x, y ,band_1[ice_sample], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-25, 10)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x, y ,band_1[ship_sample], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-25, 10)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    # show 2d pixel plots
    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    ax.imshow(band_1[ice_sample])
    ax.set_title('HH')
    ax = fig.add_subplot(1,3,2)
    ax.imshow(band_2[ice_sample])
    ax.set_title('HV')
    ax = fig.add_subplot(1,3,3)
    ax.imshow(band_3[ice_sample])
    ax.set_title('HH + HV')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    ax.imshow(band_1[ship_sample])
    ax.set_title('HH')
    ax = fig.add_subplot(1,3,2)
    ax.imshow(band_2[ship_sample])
    ax.set_title('HV')
    ax = fig.add_subplot(1,3,3)
    ax.imshow(band_3[ship_sample])
    ax.set_title('HH + HV')
    plt.show()


def start_training(x_train, y_train):
    train_size = len(x_train)
    # construct network architecture
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=(75, 75, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # define optimizer and callback function
    my_opt = Adam(lr=0.0001, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=my_opt, loss='binary_crossentropy', metrics=['accuracy'])
    earlyStopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
    # generate new data by flipping images
    batches = 32
    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow(x=x_train,
                                         y=y_train,
                                         batch_size=batches)
    # start training
    model.fit(train_generator,
          batch_size = batches,
          steps_per_epoch = train_size // batches,
          epochs = 100,
          callbacks = earlyStopping,
          verbose = 0)
    
    return model


# plot the study curve
def study_curve(his):
    acc = his.history['accuracy']
    loss = his.history['loss']
    val_acc = his.history['val_accuracy']
    val_loss = his.history['val_loss']
    epoch=range(len(acc))

    plt.figure()
    plt.plot(epoch, acc)
    plt.plot(epoch, val_acc)
    plt.title('Accuracy of Training Set and Test Set')
    plt.legend(['training', 'test'], loc='lower right')
    plt.xlabel("epoch")
    plt.savefig("/tmp/fig6.png")
    plt.show()


# plot ROC curve
def get_roc(result, y):
    fpr, tpr, threshold = roc_curve(y, result, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    str1 = 'Area = %0.2f' % (roc_auc)
    plt.text(0.4, 0.5, str1)
    plt.savefig("/tmp/fig7.png")
    plt.show()


# get results
def get_evaluation(result, y):
    tp, fp, tn, fn = 0, 0, 0, 0
    size = len(result)
    for i in range(size):
        if result[i] > 0.5:
            if y[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y[i] == 0:
                tn += 1
            else:
                fn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, precision, recall


# extract data from json
data_dir = "iceberg_data.json"
data = pd.read_json(data_dir)

# rebuild data
band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])
band_3 = (band_1 + band_2) / 2
angle = np.array(pd.to_numeric(data['inc_angle'], errors='coerce'))
# labels of images
labels = np.array(data["is_iceberg"])
# images with 3 layers
images = np.concatenate((band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis], band_3[:, :, :, np.newaxis]), axis=-1)

# plot sample images
#plot_sample(band_1, band_2, band_3, labels)

# k-fold cross-validation method
skf = StratifiedKFold(n_splits=5)
fold = 1
overall_accuracy = 0
overall_precision = 0
overall_recall = 0
for train_i, test_i in skf.split(images, labels):
    print("fold: " + str(fold))
    x_train, x_test = images[train_i], images[test_i]
    y_train, y_test = labels[train_i], labels[test_i]
    # train the model
    model = start_training(x_train, y_train)
    # predict the test set
    results = model.predict(x_test)
    accuracy, precision, recall = get_evaluation(results, y_test)
    print("accuracy: " + str(accuracy))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    overall_accuracy += accuracy
    overall_precision += precision
    overall_recall += recall
    fold += 1
# get final results
overall_accuracy /= 5
overall_precision /= 5
overall_recall /= 5
print("overall accuracy: " + str(overall_accuracy))
print("overall precision: " + str(overall_precision))
print("overall recall: " + str(overall_recall))