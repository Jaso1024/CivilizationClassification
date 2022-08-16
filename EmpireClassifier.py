from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from TextFormatter import Formatter
import numpy as np
from itertools import count

formatter = Formatter()
data, labels = formatter.get_training()
data = formatter.vectorize(data)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(data, labels)
print(decision_tree.score(data, labels))

random_forest = RandomForestClassifier()
support_vector_machine = SupportVectorClassifier()