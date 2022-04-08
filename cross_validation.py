from collections import Counter
from sklearn import metrics
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_SPLITS = 9  # 28
TEST_SIZE = 60  # 30


def create_preprocessor(pred, comp):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Standardize features by removing the mean and scaling to unit variance
        ("pca", PCA(n_components=comp))
    ])

    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, pred)
        ])


def cross_validation(dataset, predictors, classifier):
    # CROSS VALIDATION

    # n_splits is the number of subsets created
    # test_size is the number of records for each test sets
    tscv = TimeSeriesSplit(gap=0, n_splits=N_SPLITS, test_size=TEST_SIZE)

    accuracy_scores = []
    f1_scores_down = []
    f1_scores_flat = []
    f1_scores_up = []

    precision_scores_down = []
    precision_scores_flat = []
    precision_scores_up = []

    recall_scores_down = []
    recall_scores_flat = []
    recall_scores_up = []
    for train_index, test_index in tscv.split(dataset):
        print("--------------------------------------------------------------------------")
        # print("TRAIN:", train_index, "\n TEST:", test_index)

        train_dataset, test_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
        x_train, x_test = train_dataset[predictors], test_dataset[predictors]
        y_train, y_test = train_dataset[['trend']], test_dataset[['trend']]

        print("Class label for training set : ", Counter(y_train['trend']))
        print("Class label for test set : ", Counter(y_test['trend']))

        preprocessor = create_preprocessor(predictors, None)

        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor)
            , ('classifier', classifier)
        ])

        # Train the model
        pipe.fit(x_train, y_train.values.ravel())

        # Use model to make predictions
        y_pred = pipe.predict(x_test)

        # Evaluate the performance
        print("\nTraining ", classifier)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print("Accuracy on test set: ", accuracy)
        print("Metrics per class on test set:")
        print("Confusion matrix:")
        metrics.confusion_matrix(y_test, y_pred)
        print(metrics.classification_report(y_test, y_pred))

        f1 = f1_score(y_test, y_pred, average=None)
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)

        try:
            f1_scores_down.append(f1[0])
        except Exception as e:
            f1_scores_down.append(0)
            print("Exception: " + str(e))
        try:
            precision_scores_down.append(precision[0])
        except Exception as e:
            precision_scores_down.append(0)
            print("Exception: " + str(e))
        try:
            recall_scores_down.append(recall[0])
        except Exception as e:
            recall_scores_down.append(0)
            print("Exception: " + str(e))

        try:
            f1_scores_flat.append(f1[1])
        except Exception as e:
            f1_scores_flat.append(0)
            print("Exception: " + str(e))
        try:
            precision_scores_flat.append(precision[1])
        except Exception as e:
            precision_scores_flat.append(0)
            print("Exception: " + str(e))
        try:
            recall_scores_flat.append(recall[1])
        except Exception as e:
            recall_scores_flat.append(0)
            print("Exception: " + str(e))

        try:
            f1_scores_up.append(f1[2])
        except Exception as e:
            f1_scores_up.append(0)
            print("Exception: " + str(e))

        try:
            precision_scores_up.append(precision[2])
        except:
            precision_scores_up.append(0)
            print("Exception: " + str(e))
        try:
            recall_scores_up.append(recall[2])
        except Exception as e:
            recall_scores_up.append(0)
            print("Exception: " + str(e))

    print("ACCURACY OVERALL MEAN: " + str(np.mean(accuracy_scores)) + " FOR CLASSIFIER: " + str(classifier))
    print("F1 OVERALL MEAN FOR LABEL -1: " + str(np.mean(f1_scores_down)) + " / FOR LABEL 0: " + str(
        np.mean(f1_scores_flat)) + " / FOR LABEL 1: " + str(
        np.mean(f1_scores_up)) + " FOR CLASSIFIER: " + str(classifier))

    print(precision_scores_down, precision_scores_flat, precision_scores_flat)
    print(recall_scores_down, recall_scores_flat, recall_scores_up)
    print(f1_scores_down, f1_scores_flat, f1_scores_up)

    return np.mean(accuracy_scores), np.mean(f1_scores_down), np.mean(f1_scores_flat), np.mean(f1_scores_up),   \
           np.mean(precision_scores_down), np.mean(precision_scores_flat), np.mean(precision_scores_up),        \
           np.mean(recall_scores_down),  np.mean(recall_scores_flat), np.mean(recall_scores_up),                \
           accuracy_scores, f1_scores_down, f1_scores_flat, f1_scores_up