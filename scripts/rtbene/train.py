#!/usr/bin/env python3

import argparse

from numpy import mean
from numpy import std
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Random Forest
from sklearn.ensemble import RandomForestClassifier
# import SVC classifier
from sklearn import svm

from dataloader import load_dataset


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        required=True,
        help="annotations file"
    )
    parser.add_argument(
        "--output_model",
        required=True,
        help="model path"
    )
    parser.add_argument(
        "--threshold",
        default=-1,
        type=int,
        help="max number of samples of each class, -1 means all available"
    )
    return parser.parse_args()


def random_forest(X, y):
    # Cross-validation
    model = RandomForestClassifier()  # n_estimators=6, max_depth=6, random_state=0)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=1, error_score='raise')

    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    # Train a model
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, stratify=y, random_state=1)

    # model = RandomForestClassifier(n_estimators=6, max_depth=6, random_state=0, class_weight="balanced")
    model = RandomForestClassifier(class_weight="balanced")
    # fit the model on the whole dataset
    model.fit(X_train, y_train)
    # training
    y_pred = model.predict(X_train)
    # confusion matrix
    print("confusion matrix:\n", confusion_matrix(y_train, y_pred))
    # classification report
    print("classification report:\n", classification_report(y_train, y_pred))

    # testing
    y_pred = model.predict(X_test)
    print(y_pred)
    # confusion matrix
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
    # classification report
    print("classification report:\n", classification_report(y_test, y_pred))


def svm_multiclass(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, test_size=0.20, random_state=101, stratify=y)

    # rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1, class_weight="balanced").fit(X_train, y_train)
    rbf = svm.SVC(kernel='poly', degree=4, C=0.01,
                  class_weight="balanced").fit(X_train, y_train)

    # Train
    print("Training set evaluation results:")
    y_pred = rbf.predict(X_train)
    print("confusion matrix:\n", confusion_matrix(y_train, y_pred))
    print("classification report:\n", classification_report(y_train, y_pred))

    # Test
    print("Testing set evaluation results:")
    rbf_pred = rbf.predict(X_test)
    # rbf_accuracy = accuracy_score(y_test, rbf_pred)
    # rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
    # print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    # print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
    print("confusion matrix:\n", confusion_matrix(y_test, rbf_pred))
    print("classification report:\n", classification_report(y_test, rbf_pred))


def svm_gridSeach(X, y, output_model_path):
    param_grid = {
        'C': [0.01, 0.1, 1],
        'gamma': ["scale", "auto", 1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],
        'class_weight': ['balanced'], }
    # 'probability': [True]}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, test_size=0.20, random_state=101, stratify=y)

    grid = GridSearchCV(
        svm.SVC(), param_grid,
        refit="f1_micro", verbose=3,
        scoring=["f1_micro"],
        cv=StratifiedKFold(n_splits=5),
        n_jobs=1)
    grid.fit(X_train, y_train)

    print(grid.best_estimator_)

    # Train
    print("Training set evaluation results:")
    grid_predictions = grid.best_estimator_.predict(X_train)
    print("confusion matrix:\n", confusion_matrix(y_train, grid_predictions))
    print("classification report:\n",
          classification_report(y_train, grid_predictions))
    # Test
    print("Testing set evaluation results:")
    grid_predictions = grid.best_estimator_.predict(X_test)
    print("confusion matrix:\n", confusion_matrix(y_test, grid_predictions))
    print("classification report:\n",
          classification_report(y_test, grid_predictions))

    # Save
    with open(output_model_path, 'wb') as file:
        pickle.dump(grid.best_estimator_, file)


if __name__ == "__main__":
    args = parse()

    # load dataset
    X, y, _ = load_dataset(args.annotations, augment=True,
                           normalize=False, threshold=args.threshold)
    print("Data loaded")

    # Random Forest
    # random_forest(X, y)

    # SVM
    # svm_multiclass(X, y)

    # SVM - GridSearch
    svm_gridSeach(X, y, args.output_model)

    # Best Parameters
    # SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    # decision_function_shape='ovr', degree=2, gamma=0.1, kernel='rbf',
    # max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    # verbose=False)

    # Training set evaluation results:
    # confusion matrix:
    # [[240   0   0   0   0]
    # [  0 232   4   2   2]
    # [  1  10 223   0   6]
    # [  2   9   0 225   4]
    # [  4  12   3   2 219]]
    # classification report:
    #             precision    recall  f1-score   support
    #
    #         0       0.97      1.00      0.99       240
    #         1       0.88      0.97      0.92       240
    #         2       0.97      0.93      0.95       240
    #         3       0.98      0.94      0.96       240
    #         4       0.95      0.91      0.93       240
    #
    #     accuracy                           0.95      1200
    #    macro avg       0.95      0.95      0.95      1200
    # weighted avg       0.95      0.95      0.95      1200

    # Testing set evaluation results:
    # confusion matrix:
    # [[59  0  0  1  0]
    # [ 0 55  2  2  1]
    # [ 0  1 56  0  3]
    # [ 0  6  0 53  1]
    # [ 4  2  6  0 48]]
    # classification report:
    #             precision    recall  f1-score   support
    #
    #         0       0.94      0.98      0.96        60
    #         1       0.86      0.92      0.89        60
    #         2       0.88      0.93      0.90        60
    #         3       0.95      0.88      0.91        60
    #         4       0.91      0.80      0.85        60
    #
    #     accuracy                           0.90       300
    #    macro avg       0.90      0.90      0.90       300
    # weighted avg       0.90      0.90      0.90       300

# only eyelids distance
