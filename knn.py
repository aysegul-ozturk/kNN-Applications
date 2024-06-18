import pandas as pd
import numpy as np
def find_distance_matrix(matrix1, matrix2, method):
    if method == 'euclidean':
        return np.sqrt(((matrix1 - matrix2) ** 2).sum(axis=1))
    elif method == 'manhattan':
        return np.abs(matrix1 - matrix2).sum(axis=1)
    elif method == 'chebyshev':
        return np.max(np.abs(matrix1 - matrix2), axis=1)

def find_distance_row(row1, row2, method):
    if method == 'euclidean':
        return np.sqrt(np.sum((row1 - row2) ** 2))
    elif method == 'manhattan':
        return np.sum(abs(row1 - row2))
    elif method == 'chebyshev':
        return np.max(abs(row1 - row2))

def knn(existing_data, test_data, k, distance_method, re_training, distance_threshold, weighted_voting):
    train_labels = existing_data['label'].to_numpy()
    existing_attributes = existing_data.drop('label', axis='columns').to_numpy()
    test_attributes = test_data.drop('label', axis='columns').to_numpy()

    estimates_arr = []

    for test_record in test_attributes:
        distances = find_distance_matrix(existing_attributes, test_record, distance_method)

        if distance_threshold is None:
            in_distance_labels = train_labels
        else:
            in_distance_labels = train_labels[distances <= distance_threshold]
            distances = distances[distances <= distance_threshold]

        if in_distance_labels.size == 0:
            in_distance_labels.size
            estimates_arr.append(np.nan)
            continue

        k_neighbor_labels = in_distance_labels[np.argsort(distances)[:k]].astype(int)

        if weighted_voting is False:
            estimate = np.bincount(k_neighbor_labels).argmax()
        else:
            weights = 1 / (distances[np.argsort(distances)[:k]] + 1e-3)
            weighted_votes = np.zeros(np.max(k_neighbor_labels) + 1)
            np.add.at(weighted_votes, k_neighbor_labels, weights)
            estimate = np.argmax(weighted_votes)

        estimates_arr.append(estimate)

        if re_training is True:
            train_labels = np.append(train_labels, estimate)
            existing_attributes = np.vstack([existing_attributes, test_record])

    return pd.Series(estimates_arr, index=test_data.index)


def fill_missing_features(existing_data, test_data, k, distance_method, distance_threshold,
                          weighted_voting):

    existing_attributes = existing_data.drop(columns=['label']).values
    missing_attributes = test_data.drop(columns=['label'])

    for attr in missing_attributes.columns:
        missing_indices = test_data[attr].isna()

        for missing_index in test_data[missing_indices].index:
            test_sample = test_data.loc[missing_index].drop('label').values

            distances = np.array([
                find_distance_row(
                    np.nan_to_num(test_sample, nan=0),
                    np.nan_to_num(existing_attributes[i], nan=0),
                    distance_method
                )
                for i in range(len(existing_attributes))
            ])

            if distance_threshold is None:
                in_distance_indices = np.arange(len(existing_attributes))
            else:
                in_distance_indices = np.where(distances <= distance_threshold)[0]
                if in_distance_indices.size == 0:
                    continue

            k_neighbor_indices = in_distance_indices[np.argsort(distances[in_distance_indices])[:k]]

            if weighted_voting is True:
                weights = 1 / (distances[k_neighbor_indices] + 1e-5)
                estimated_value = np.average(
                    existing_attributes[k_neighbor_indices, test_data.columns.get_loc(attr) - 1],
                    weights=weights)
            else:
                estimated_value = np.mean(
                    existing_attributes[k_neighbor_indices, test_data.columns.get_loc(attr) - 1])

            test_data.at[missing_index, attr] = estimated_value

    return test_data
