from typing import Dict, Tuple, List
import psb2
# import autokeras as aks
import numpy as np
import sklearn as skl

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import autosklearn.classification as askc

# size constants for use in declaring our askl classifier
MB = 1
GB = 1024 * MB
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR


def get_dataset(size=500_000) -> Tuple:
    X, y = [], []
    for problem in psb2.PROBLEMS:
        # we don't load in anything split between test and train because
        # we plan on nesting stratified K fold validation in main
        (train_data, _test_data) = psb2.fetch_examples(
            "../data", problem, size, 0, seed=1)
        X.extend(train_data)
        y.extend([problem for i in range(size)])

    return (np.array(X), np.array(y))


def extract_features(example: Dict) -> Dict:
    try:
        features = {
            'input_count': sum('input' in k for k in example),
            'input_type': type(example['input1']),
            'output_count': sum('output' in k for k in example),
            'output_type': type(example['output1']),
            'input_length': 1 if (type(example['input1']) == type(1.2) or type(example['input1']) == type(1)) else len(example['input1'])
        }
        return features
    except:
        print(example)


def one_hot_encode(features: Dict) -> List:
    feature_list = [features['input_count'],
                    features['input_length'],
                    features['output_count']]
    # int string list float bool
    for feature in ['input_type', 'output_type']:
        types = [1, 'a', [1, 2], 1.2, False]
        for t in types:
            if features[feature] == type(t):
                feature_list.append(1)
            else:
                feature_list.append(0)

    return feature_list


def get_models(X: List, y: List) -> Dict:
    models = {}
    models['gbc'] = GradientBoostingClassifier().fit(X, y)
    models['rf'] = RandomForestClassifier().fit(X, y)
    models['dt'] = DecisionTreeClassifier().fit(X, y)
    models['knn'] = KNeighborsClassifier().fit(X, y)
    models['svm'] = SVC().fit(X, y)
    models['log'] = LogisticRegression().fit(X, y)
    return models


def main():
    print(
        "Loading Data. This may take a moment if the entire dataset hasn't been cached.")
    X, y = get_dataset(size=50_000)

    X, y = skl.utils.shuffle(X, y)

    print('Beginning preprocessing')
    X = np.array(
        list(map(one_hot_encode, map(extract_features, X))))
    print('Finished preprocessing')

    iteration = 0

    # modify the number of folds for cross validation
    accuracy = {}
    hyper = StratifiedKFold(n_splits=5)
    for data_idx, val_idx in hyper.split(X, y):
        X_data, X_val = X[data_idx], X[val_idx]
        y_data, y_val = y[data_idx], y[val_idx]

        models = []

        fold = StratifiedKFold(n_splits=5)
        for train_idx, test_idx in fold.split(X_data, y_data):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            print('Beginning model training')
            automl = askc.AutoSklearnClassifier(
                include = {
                    'classifier': ["random_forest", "gradient_boosting", "k_nearest_neighbors", "adaboost", "libsvm_svc", "decision_tree"],
                    'feature_preprocessor': ["no_preprocessing"]
                },
                time_left_for_this_task = 1 * HOUR,
                per_run_time_limit = 20 * MINUTE,
                seed = iteration,
                n_jobs = 5,
                memory_limit = 12 * GB,
            )

            automl.fit(X_train, y_train)
            prediction = automl.predict(X_test)
            acc = accuracy_score(prediction, y_test)
            models = get_models(X_train, y_train)

            scores = {}
            print('Scoring models')
            for name in tqdm(models.keys()):
                scores[name] = models[name].score(X_test, y_test)
            with open(f'results_{iteration}.out', "w+") as f:
                f.write(automl.sprint_statistics())
                f.write("\n\n")
                f.write(str(automl.show_models()))
                f.write(f'accuracy: {acc}')
                f.write(f'scores: {scores}')
            print(automl.sprint_statistics())
            accuracy[iteration] = acc
            iteration += 1

    print(accuracy)


if __name__ == "__main__":
    main()
