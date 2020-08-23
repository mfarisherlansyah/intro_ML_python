from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_breast_cancer

import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type = str, default = "knn", help = "type of python machine learning to use")
args = vars(ap.parse_args())

# Define the dictionary of models our script can use
models = {
	"knn": KNeighborsClassifier(n_neighbors = 1),
	"naive_bayes": GaussianNB(),
	"logit": LogisticRegression(solver = "lbfgs", multi_class = "auto"),
	"svm": SVC(kernel = "rbf", gamma = "auto"),
	"decision_tree": DecisionTreeClassifier(),
	"random_forest": RandomForestClassifier(n_estimators = 100),
	"mlp": MLPClassifier(),
	"perceptron": Perceptron(max_iter = 50)
}

# Load the iris data and perform a training and testing split
print("[INFO] loading data...")

dataset = load_breast_cancer()
(trainx, testx, trainy, testy) = train_test_split(dataset.data, dataset.target, random_state = 3, test_size = 0.25)

# Train the model
print("[INFO] using {} model".format(args["model"]))
model = models[args["model"]]
model.fit(trainx, trainy)

print("[INFO] evaluating...")
predictions = model.predict(testx)
print(classification_report(testy, predictions, target_names = dataset.target_names))