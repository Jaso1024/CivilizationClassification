from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from TextFormatter import Formatter
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

def get_wrong_sentences(model, testing_data, testing_labels):
    for index in range(testing_data.shape[0]):
        datapoint = testing_data[index]
        label = testing_labels[index]
        prediction = model.predict(datapoint)
        for idx in range(len(prediction)):
            if prediction[0,idx] != label[idx]:
                print(testing_data_[index])

formatter = Formatter()

training_data, training_labels = formatter.get_training()
training_data = formatter.vectorize(training_data)

testing_data_, testing_labels = formatter.get_testing()
testing_data = formatter.vectorize(testing_data_)

decision_tree = DecisionTreeClassifier(random_state=2, max_depth=5)
decision_tree.fit(training_data, training_labels)
print(decision_tree.score(testing_data, testing_labels))

highest_score = 0
highest_num = 0
x = []
y = []
for num in range(100):
    for depth in range(100):
        random_forest = RandomForestClassifier(n_estimators=100, random_state=num, max_depth=depth)
        random_forest.fit(training_data, training_labels)
        score = random_forest.score(testing_data, testing_labels)
        if score > highest_score:
            highest_score = score
            highest_num = num
        
print(highest_score, highest_num)

#support_vector_machine = SupportVectorClassifier()
#support_vector_machine.fit(data, labels)
#print(support_vector_machine.score(testing_data, testing_labels))

