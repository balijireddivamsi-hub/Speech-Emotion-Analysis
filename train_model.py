import os
import pickle
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

DATASET_PATH = "dataset"
MODEL_DIR = "artifacts"

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def extract_features(path):
    audio, sr = librosa.load(
        path, sr=22050, duration=3, offset=0.5
    )
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=40
    )
    return np.mean(mfcc.T, axis=0)


def main():
    features, labels, actors = [], [], []

    for root, _, files in os.walk(DATASET_PATH):
        for filename in files:
            if not filename.endswith(".wav"):
                continue

            parts = filename.split("-")

            # RAVDESS speech-only recordings
            if len(parts) != 7:
                continue
            if parts[0] != "03" or parts[1] != "01":
                continue

            emotion_code = parts[2]
            if emotion_code not in EMOTIONS:
                continue

            path = os.path.join(root, filename)

            try:
                features.append(extract_features(path))
                labels.append(EMOTIONS[emotion_code])
                actors.append(parts[6].split(".")[0])
            except Exception as exc:
                print(f"Skipping {filename}: {exc}")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    actors = np.asarray(actors)

    if len(y) == 0:
        raise ValueError(
            "No recordings found. Check the dataset folder."
        )

    print("Recordings:", len(y))
    print("Feature shape:", X.shape)

    # Keep actors separate to reduce speaker leakage.
    train_actors, test_actors = train_test_split(
        np.unique(actors),
        test_size=0.25,
        random_state=42
    )
    train_mask = np.isin(actors, train_actors)
    test_mask = np.isin(actors, test_actors)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Input(shape=(40,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(len(encoder.classes_), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train_enc,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    loss, accuracy = model.evaluate(
        X_test, y_test_enc, verbose=0
    )
    predictions = np.argmax(
        model.predict(X_test, verbose=0), axis=1
    )

    print(f"Unseen-actor test accuracy: {accuracy:.2%}")
    print(classification_report(
        y_test_enc,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    ))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/emotion_model.keras")

    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{MODEL_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print("Training complete. Files saved in artifacts/.")


if __name__ == "__main__":
    main()