"""Train a TF-IDF and LinearSVC fake-news classifier."""

import argparse
from pathlib import Path

import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate import evaluate
from preprocess import CLASS_NAMES, DEFAULT_DATA_SOURCE, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    df = load_dataset(args.data)
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["class"],
        stratify=df["class"], test_size=args.test_size, random_state=42,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7,
                                   stop_words="english")),
        ("svm", LinearSVC(dual=False, max_iter=100000)),
    ])

    param_grid = {
        "svm__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "svm__penalty": ["l1", "l2"],
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(x_train_text, y_train)

    print("Best hyperparameters:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test_text)
    y_score = best_model.decision_function(x_test_text)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    evaluate(
        y_test, y_pred, args.out_dir, "svm", y_score=y_score,
        classes=CLASS_NAMES,
    )

    model_path = args.out_dir / "svm_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved model + vectorizer pipeline to {model_path}")


if __name__ == "__main__":
    main()
