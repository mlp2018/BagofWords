#One-Dimensional Convolutional Neural Network Model 
#designed to honor the spatial structure in image data whilst being robust to the position and #orientation of #learned objects in the scene.
""" can be used on sequences, such as the one-dimensional sequence of words in a movie review. The same properties that make the CNN model attractive for learning to recognize objects in images can help to learn structure in paragraphs of words, namely the techniques invariance to the specific position of features."""

# CNN for the IMDB problem
import os
import sys
from pathlib import Path
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import regularizers
from keras import optimizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd

def _get_current_file_dir() -> Path:
    """Returns the directory of the script."""
    try:
        return Path(os.path.realpath(__file__)).parent
    except(NameError):
        return Path('')


# Project root directory, i.e. the github repo directory.
_PROJECT_ROOT = _get_current_file_dir() / '..'

# Where to find the data files.
_DATA_FILE = {
    'labeled':
        _PROJECT_ROOT / 'data' / 'labeledTrainData.tsv',
    'unlabeled':
        _PROJECT_ROOT / 'data' / 'unlabeledTrainData.tsv',
    'test':
        _PROJECT_ROOT / 'data' / 'testData.tsv',
    'result':
        _PROJECT_ROOT / 'results' / 'PredictionCNNM.csv'
}


def read_data(keyword: str) -> pd.DataFrame:
    """
    Reads data from the tsv file corresponding to ``_DATA_FILE[keyword]``
    and returns it as a :py:class:`panda.DataFrame`.
    """
    return pd.read_csv(_DATA_FILE[keyword], header=0, delimiter='\t',
                       quoting=3)


##definition for plotting the model
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train_set', 'validation_set'], loc='best')
    plt.show()


X_train_data = read_data('labeled')
X_test_data = read_data('test')



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


# create the model
model = Sequential()
model.add(Embedding(top_words, 100, input_length=max_words))
model.add(Conv1D(filters=90, kernel_size=6, padding='same', activation='relu', use_bias = True))
model.add(MaxPooling1D(pool_size=500, strides = 2))
model.add(Flatten())
#model.add(BatchNormalization(center=True, scale=True))
model.add(Dense(250, activation='relu', use_bias = True))
#Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting. the network becomes less sensitive to the specific weights of neurons.
model.add(Dropout(0.1)) 
model.add(Dense(1, activation='sigmoid', use_bias= True)) 
#compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0021), metrics=['accuracy']) #TRY OPTIMIZERS
print(model.summary())


#Stop training when a monitored quantity has stopped improving.
#Patience how much should it be tolerable.
earlystop = EarlyStopping(patience = 1)
callbacks_list = [earlystop]
# Fit the model ( a number of epochs means how many times you go through your training set.)
model_info = model.fit(X_train, y_train, epochs=2, batch_size=128, verbose=2, callbacks=callbacks_list)
#calculate predictions
predictions = model.predict(X_test)
# round predictions
result = [round(x[0]) for x in predictions]

pd.DataFrame(data={"id": X_test["id"], "sentiment": result}) \
	.to_csv(_DATA_FILE['result'], index=False, quoting=3)
#preds = pd.DataFrame(rounded, columns=['predictions']).to_csv('predictionCNN.csv')
#np.savetxt('~/BagofWords/results/CNNMpreds.csv', rounded, delimiter=',')
# plot model history
plot_model_history(model_info)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


