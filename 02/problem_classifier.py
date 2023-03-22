from typing import Dict, Tuple, List
import psb2
import autokeras as aks
import numpy as np
import sklearn as skl

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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
            "../data", problem, size, 0)
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


def main():
    print(
        "Loading Data. This may take a moment if the entire dataset hasn't been cached.")
    X, y = get_dataset(size=10_000)

    X, y = skl.utils.shuffle(X, y)

    print('Beginning preprocessing')
    X = np.array(
        list(map(one_hot_encode, map(extract_features, X))))
    # X_test = list(map(one_hot_encode, map(extract_features, X_test)))
    print('Finished preprocessing')

    iteration = 0

    # modify the number of folds for cross validation
    accuracy = {}
    hyper = StratifiedKFold(n_splits=4)
    for data_idx, val_idx in hyper.split(X, y):
        X_data, X_val = X[data_idx], X[val_idx]
        y_data, y_val = y[data_idx], y[val_idx]

        models = []

        fold = StratifiedKFold(n_splits=3)
        for train_idx, test_idx in fold.split(X_data, y_data):
            # X_train, X_test = X[train_idx], X[test_idx]
            # y_train, y_test = y[train_idx], y[test_idx]

            print('Beginning model training')
            automl = aks.StructuredDataClassifier(
                max_trials=25,
                seed=1,
                objective='val_accuracy',
                project_name=f'problem_classifier_{iteration}'
            )

            automl.fit(X_data, y_data)
            automl.export_model()
            models.append(automl)
            iteration += 1

        for model in models:
            predictions = model.predict(X_val)
            accuracy[int(iteration / 3)] = accuracy_score(predictions, y_val)

    print(accuracy)


if __name__ == "__main__":
    main()
