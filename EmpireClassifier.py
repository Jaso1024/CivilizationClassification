from gc import callbacks
from lib2to3.pgen2.literals import test
from unicodedata import bidirectional
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, SimpleRNN, InputLayer, Embedding, Input, Bidirectional, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax, RMSprop
from keras.callbacks import EarlyStopping


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from TextFormatter import Formatter

def get_wrong_sentences(model, testing_data, testing_labels):
    for index in range(testing_data.shape[0]):
        datapoint = testing_data[index]
        label = testing_labels[index]
        prediction = model.predict(datapoint)
        for idx in range(len(prediction)):
            if prediction[0,idx] != label[idx]:
                print(testing_data_[index])

formatter = Formatter()
data = formatter.get_data()
training_data_text, testing_data_text, training_labels, testing_labels = train_test_split(*data, test_size=.5, random_state=0)
training_data = formatter.vectorize(training_data_text)
testing_data = formatter.vectorize(testing_data_text)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)
num_uniques = formatter.get_unique_words(data[0])

# Decision Tree
best_depth = 24
best_features = 433
decision_tree = DecisionTreeClassifier(random_state=2, max_depth=24, max_features=best_features)
decision_tree.fit(training_data, training_labels)
decision_tree_score = decision_tree.score(testing_data, testing_labels)

# Random Forest
best_depth = 56
best_features = 87
random_forest = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=best_depth, max_features=best_features)
random_forest.fit(training_data, training_labels)
random_forest_score = random_forest.score(testing_data, testing_labels)

#Sklearn's Support vector machine dosen't support one-hot encoded labels
# Support Vector Machine
best_c = 0.1
best_kernel = "linear"
support_vector_machine = SupportVectorClassifier(kernel=best_kernel, C=best_c)
support_vector_machine.fit(training_data, formatter.linear_encode(training_labels))
support_vector_machine_score = support_vector_machine.score(testing_data, formatter.linear_encode(testing_labels))



# Bi-directional LSTM
formatted_text, max_len = formatter.tensorflow_format_text(data[0], num_uniques)
training_data, testing_data, training_labels, testing_labels = train_test_split(formatted_text, data[1], test_size=.33, random_state=42)

rnn = Sequential()
rnn.add(Embedding(num_uniques, 128, input_length=max_len))
rnn.add(Bidirectional(LSTM(64)))
rnn.add(Dense(256, activation="relu"))
rnn.add(Dropout(0.5))
rnn.add(Dense(128))
rnn.add(Dense(3, activation="sigmoid"))

#es = EarlyStopping(monitor="accuracy", min_delta=0.0001, restore_best_weights=True)
rnn.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=["accuracy"])


rnn.fit(training_data, np.array(training_labels), epochs=1, verbose=1, batch_size=100)
print(rnn.evaluate(testing_data, np.array(testing_labels)))






