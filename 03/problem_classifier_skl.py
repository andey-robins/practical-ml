from typing import Dict, Tuple, List
import psb2
import math
# import autokeras as aks
import numpy as np
import sklearn as skl

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import autosklearn.classification as askc
from joblib import Parallel, delayed

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


def cross_validation_train(features):
    return askc.AutoSklearnClassifier(
        include = features['include'],
        time_left_for_this_task = features['time_left'],
        per_run_time_limit = features['time_limit'],
        seed = 1,
        n_jobs = 5,
        memory_limit = 8 * GB,
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": 4},
    )


def get_features_from_iteration(i):
    CLS_SUBSETS = [
        ["random_forest", "k_nearest_neighbors"],
        ["random_forest", "libsvm_svc"],
        ["gradient_boosting", "k_nearest_neighbors"],
        ["gradient_boosting", "libsvm_svc"],
        ["k_nearest_neighbors", "libsvm_svc"],
        ["k_nearest_neighbors", "decision_tree"],
        ["adaboost", "libsvm_svc"],
        ["random_forest"],
        ["gradient_boosting"],
        ["k_nearest_neighbors"],
        ["libsvm_svc"],
        ["decision_tree"]
    ]
    TIME_SUBSETS = [
        (5 * MINUTE, 10 * MINUTE),
        (5 * MINUTE, 30 * MINUTE),
        (10 * MINUTE, 30 * MINUTE)
    ]
    PREPROCESSING = [
        'no_preprocessing',
        'kitchen_sinks',
        'feature_agglomeration',
    ]
    classifier_idx = i % len(CLS_SUBSETS)
    time_idx = (i // len(CLS_SUBSETS)) % 3
    preprocess = i // (len(CLS_SUBSETS) * 3)
    return [
        CLS_SUBSETS[classifier_idx],
        TIME_SUBSETS[time_idx],
        PREPROCESSING[preprocess]
    ]


def worker(iteration, fold, X_train, y_train, X_val, y_val):
    print(f'Starting Worker -- iteration: {iteration}, fold: {fold}')
    feature_set = get_features_from_iteration(iteration)

    features = {
        'include': {
            'classifier': feature_set[0],
            'feature_preprocessor': [feature_set[2]]
        },
        'time_left': feature_set[1][1],
        'time_limit': feature_set[1][0]
    }

    automl = cross_validation_train(features)

    automl.fit(X_train, y_train)
    prediction = automl.predict(X_val)
    acc = accuracy_score(prediction, y_val)
    accuracy[iteration + fold*(12 * 3 * 3)] = acc
    with open(f'res/{fold}/results_{iteration}.out', "w") as f:
        f.write(automl.sprint_statistics())
        f.write("\n\n")
        f.write(str(automl.show_models()))
        f.write(f'\naccuracy: {acc}\n')
        f.write(f'{feature_set}\n')

def main():
    print(
        "Loading Data. This may take a moment if the entire dataset hasn't been cached.")
    X, y = get_dataset(size=50_000)

    X, y = skl.utils.shuffle(X, y)
    
    print('Extracting Features')
    X = np.array(
        list(map(one_hot_encode, map(extract_features, X))))
    print('Finished Feature Extraction')
        
    fold = 0

    # modify the number of folds for cross validation
    hyper = StratifiedKFold(n_splits=4)
    for data_idx, val_idx in hyper.split(X, y):
        X_train, X_val = X[data_idx], X[val_idx]
        y_train, y_val = y[data_idx], y[val_idx]
        
        Parallel(n_jobs=8)(delayed(worker)(iteration, fold, X_train, y_train, X_val, y_val) for iteration in range(12 * 3 * 3))
            
        fold += 1

    print(accuracy)

accuracy = {}

if __name__ == "__main__":
    main()
