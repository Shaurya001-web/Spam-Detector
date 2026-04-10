from pathlib import Path

from model_backend import predict_spam, train_spam_model


def main() -> None:
    csv_path = Path(__file__).parent / "combined_dataset.csv"
    artifacts = train_spam_model(csv_path)

    test_message = "Congratulations! You won a lottery prize. Click now."
    label, confidence = predict_spam(test_message, artifacts)

    print(f"Accuracy: {artifacts.accuracy:.4f}")
    print(f"Sample prediction: {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()
