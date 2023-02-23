from typing import Dict, Tuple, List
import psb2

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def get_dataset(size=10_000, test_split=0.2) -> Tuple:
    X_train, y_train = [], []
    X_test, y_test = [], []
    for problem in psb2.PROBLEMS:
        (train_data, test_data) = psb2.fetch_examples("./data", problem, int(size * 1 -
                                                      test_split), int(size * test_split))
        X_train.extend(train_data)
        y_train.extend([problem for i in range(int(size * 1 - test_split))])
        X_test.extend(test_data)
        y_test.extend([problem for i in range(int(size * test_split))])

    return (X_train, y_train, X_test, y_test)


def extract_features(example: Dict) -> Dict:
    features = {
        'input_count': sum('input' in k for k in example),
        'input_type': type(example['input1']),
        'output_count': sum('output' in k for k in example),
        'output_type': type(example['output1']),
        'input_length': 1 if (type(example['input1']) == type(1.2) or type(example['input1']) == type(1)) else len(example['input1'])
    }
    return features


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
    # modify the number of folds for cross validation
    accuracy = {}
    for fold in range(10):
        print(f'Beginning fold {fold+1}/10')
        print(
            "Loading Data. This may take a moment if the entire dataset hasn't been cached.")
        (X_train, y_train, X_test, y_test) = get_dataset()

        # preprocessing
        print('Beginning preprocessing')
        X_train = list(map(one_hot_encode, map(extract_features, X_train)))
        X_test = list(map(one_hot_encode, map(extract_features, X_test)))
        print('Finished preprocessing')

        print('Beginning model training')
        models = get_models(X_train, y_train)

        scores = {}
        print('Scoring models')
        for name in tqdm(models.keys()):
            scores[name] = models[name].score(X_test, y_test)
        accuracy[fold+1] = scores

    print(accuracy)


if __name__ == "__main__":
    main()
