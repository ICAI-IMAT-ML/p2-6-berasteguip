from sklearn.metrics import accuracy_score
import numpy as np

def cross_validation(model, X, y, nFolds):
    """
    Perform cross-validation on a given machine learning model to evaluate its performance.

    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.

    Parameters:
    - model: scikit-learn-like estimator
        The machine learning model to be evaluated. This model must implement the .fit() and .score() methods
        similar to scikit-learn models.
    - X: array-like of shape (n_samples, n_features)
        The input features to be used for training and testing the model.
    - y: array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression) for the input samples.
    - nFolds: int
        The number of folds to use for cross-validation. If set to -1, LOO cross-validation is performed.

    Returns:
    - mean_score: float
        The mean score across all cross-validation folds.
    - std_score: float
        The standard deviation of the scores across all cross-validation folds, indicating the variability
        of the score across folds.

    Example:
    --------
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_classification

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Initialize a kNN model
    model = KNeighborsClassifier(n_neighbors=5)

    # Perform 5-fold cross-validation
    mean_score, std_score = cross_validation(model, X, y, nFolds=5)

    print(f'Mean CV Score: {mean_score}, Std Deviation: {std_score}')
    """
    if nFolds == -1:
        # Implement Leave One Out CV
        nFolds = X.shape[0]

    fold_size = int(np.floor(X.shape[0] / nFolds))
    accuracy_scores = []

    for i in range(nFolds):
        # Calculate start and end indices for the validation set
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < nFolds - 1 else X.shape[0]
        valid_indices = list(range(start_idx, end_idx))

        # Generate training indices by excluding validation indices
        train_indices = list(set(range(X.shape[0])) - set(valid_indices))

        # Split the dataset into training and validation
        X_train, X_valid = X[train_indices, :], X[valid_indices, :]
        y_train, y_valid = y[train_indices], y[valid_indices]

        # Train the model with the training set
        model.fit(X_train, y_train)

        # Calculate the accuracy of the model with the validation set
        accuracy_scores.append(accuracy_score(y_valid, model.predict(X_valid)))

    # Return the mean and standard deviation of the accuracy_scores
    return np.mean(accuracy_scores), np.std(accuracy_scores)
