from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Concatenate, Embedding, Input, Bidirectional, Dropout
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from TextFormatter import Formatter

def evaluate(predictions, labels):
    num_correct = 0
    for prediction, label in zip(predictions, labels):
        correct=True
        for idx in range(len(prediction)):
            if np.array(prediction).flatten()[idx] != label[idx]:
                correct=False
        if correct:    
            num_correct += 1
    return num_correct/len(predictions)

def svm_eval(predictions, labels):
    num_correct = 0
    for prediction, label in zip(predictions, labels):
        if int(prediction) == int(label):
            num_correct += 1
    return num_correct/len(predictions)

formatter = Formatter(min_sentence_length=1)
data = formatter.get_data()
training_data_text, testing_data_text, training_labels, testing_labels = train_test_split(*data, test_size=.1, random_state=0)
training_data = formatter.vectorize(training_data_text)
testing_data = formatter.vectorize(testing_data_text)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)
num_uniques = formatter.get_unique_words(data[0])

# Decision Tree
best_depth = 24
best_features = int(num_uniques * 0.1577)
decision_tree = DecisionTreeClassifier(random_state=2, max_depth=best_depth, max_features=best_features)
decision_tree.fit(training_data, training_labels)
decision_tree_predictions = decision_tree.predict(testing_data)
decision_tree_score = evaluate(decision_tree_predictions, testing_labels)


# Random Forest
best_depth = 56
best_features = num_uniques
random_forest = RandomForestClassifier(n_estimators=31, random_state=0, max_depth=best_depth, max_features=num_uniques)
random_forest.fit(training_data, training_labels)
random_forest_score = random_forest.score(testing_data, testing_labels)

# Support Vector Machine
best_c = 0.1
best_kernel = "linear"
support_vector_machine = SupportVectorClassifier(kernel=best_kernel, C=best_c)
support_vector_machine.fit(training_data, formatter.linear_encode(training_labels))
support_vector_machine_predictions = support_vector_machine.predict(testing_data)
support_vector_machine_score = svm_eval(support_vector_machine_predictions, formatter.linear_encode(testing_labels))

# Bi-directional LSTM
formatted_text, max_len = formatter.tensorflow_format_text(data[0], num_uniques)
training_data, testing_data, training_labels, testing_labels = train_test_split(formatted_text, data[1], test_size=.33, random_state=42)

lstm = Sequential()
lstm.add(Embedding(num_uniques, 128))
lstm.add(Bidirectional(LSTM(64)))
lstm.add(Dense(256, activation="relu"))
lstm.add(Dropout(0.5))
lstm.add(Dense(128))
lstm.add(Dense(4, activation="sigmoid"))
lstm.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
lstm.fit(training_data, np.array(training_labels), epochs=5, verbose=1, batch_size=100, callbacks=[])
lstm_score = lstm.evaluate(testing_data, np.array(testing_labels), verbose=1)
# Ensemble moddeling with stacking


print("Decision tree:", decision_tree_score)
print("Random forest:", random_forest_score)
print("SVM:", support_vector_machine_score)
print("LSTM:", lstm_score)




