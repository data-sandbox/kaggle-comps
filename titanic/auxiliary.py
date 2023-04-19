"""Functions used across notebooks in one centralized source
of truth for improved maintenance"""

from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def fit_model(X, y, transformer, estimator, param_grid=None):
    """Fit classifier model"""

    pipe = Pipeline([('transformer', transformer),
                    ('scaler', MaxAbsScaler()),
                    ("est", estimator)
                     ])

    if param_grid:
        model_gs = GridSearchCV(pipe, cv=3, param_grid=param_grid,
                                verbose=True)
        model_gs.fit(X, y)
        return model_gs
    else:
        pipe.fit(X, y)
        return pipe


def predict_model(X, y, model):
    predictions = model.predict(X)
    print(classification_report(y, predictions))

    return
