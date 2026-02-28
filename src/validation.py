import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

from src.config import CV_N_SPLITS, RANDOM_SEED


def cross_validate(model, X, y, n_splits=CV_N_SPLITS, random_seed=RANDOM_SEED):
    """
    KFold кросс-валидация для регрессии.
    Возвращает RMSE по фолдам (чем меньше, тем лучше).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        preds = model_clone.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)

    scores = np.array(scores)
    print(f"RMSE по фолдам: {scores.round(4)}")
    print(f"Среднее: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores
