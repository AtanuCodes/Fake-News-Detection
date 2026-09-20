"""Train an LSTM fake-news classifier."""

import argparse
import json
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from evaluate import evaluate
from preprocess import CLASS_NAMES, DEFAULT_DATA_SOURCE, load_dataset

MAX_WORDS = 5000
MAX_LEN = 500
EMBEDDING_DIM = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    df = load_dataset(args.data)
    utils.set_random_seed(42)

    train_text, test_text, y_train_full, y_test = train_test_split(
        df["clean_text"], df["class"], stratify=df["class"],
        test_size=args.test_size, random_state=42,
    )
    train_text, val_text, y_train, y_val = train_test_split(
        train_text, y_train_full, stratify=y_train_full,
        test_size=0.20, random_state=42,
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(train_text)
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_text), maxlen=MAX_LEN)
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_text), maxlen=MAX_LEN)
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=MAX_LEN)

    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM),
        LSTM(64),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        x_train, y_train,
        epochs=args.epochs, batch_size=args.batch_size,
        validation_data=(x_val, y_val), callbacks=[early_stopping],
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.4f}")

    y_score = model.predict(x_test, verbose=0).ravel()
    y_pred = (y_score > 0.5).astype(int)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    evaluate(
        y_test, y_pred, args.out_dir, "lstm", y_score=y_score,
        classes=CLASS_NAMES,
    )

    model.save(args.out_dir / "lstm_model.keras")
    with open(args.out_dir / "lstm_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    (args.out_dir / "lstm_config.json").write_text(
        json.dumps({"max_words": MAX_WORDS, "max_len": MAX_LEN}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model, tokenizer, and config to {args.out_dir}")


if __name__ == "__main__":
    main()
