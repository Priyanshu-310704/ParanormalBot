from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from .dataset_loader import load_dataset
import joblib
from pathlib import Path


def train_model():
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    data = load_dataset()
    texts = [item['text'] for item in data]
    labels = [1 if item['label'] == 'paranormal' else 0 for item in data]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save models
    joblib.dump(vectorizer, models_dir / "vectorizer.joblib")
    joblib.dump(model, models_dir / "classifier.joblib")

    print(f"✅ Model trained! Accuracy: {model.score(X_test, y_test):.2f}")
    print(f"Models saved to {models_dir}")


if __name__ == "__main__":
    train_model()