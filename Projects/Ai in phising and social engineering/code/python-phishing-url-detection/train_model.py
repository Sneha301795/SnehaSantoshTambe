import pandas as pd
from pycaret.classification import setup, compare_models, save_model, load_model, predict_model
import urllib.parse


class PhishingDetector:
    def __init__(self, model_path=None):
        """
        Initialize the PhishingDetector.
        If model_path is provided, loads an existing model, otherwise expects training.
        """
        self.model = None
        if model_path:
            self.model = load_model(model_path)

    def train_model(self, csv_path='phishing.csv', save_path='model/phishing_detection_model'):
        """
        Train a phishing detection model from a CSV file.

        Args:
            csv_path (str): Path to the training CSV file
            save_path (str): Path to save the trained model
        """
        # Load the dataset
        df = pd.read_csv(csv_path)

        # Setup PyCaret environment
        clf = setup(data=df, target='Index', session_id=123, verbose=False)

        # Compare models and select the best one
        best_model = compare_models(verbose=False)

        # Save the model
        save_model(best_model, save_path)
        self.model = best_model

        print(f"Model trained and saved to {save_path}")
        return best_model

    def predict_url(self, url):
        """
        Predict if a URL is phishing (1) or legitimate (0).

        Args:
            url (str): The URL to check

        Returns:
            dict: Prediction and probability scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Either train a new model or load an existing one.")

        # Extract features from the URL (simplified example)
        features = self._extract_url_features(url)
        features_df = pd.DataFrame([features])

        # Make prediction
        prediction = predict_model(self.model, data=features_df)

        return {
            'url': url,
            'prediction': 'phishing' if prediction['Label'][0] == 1 else 'legitimate',
            'phishing_probability': prediction['Score'][0],
            'legitimate_probability': 1 - prediction['Score'][0]
        }

    def _extract_url_features(self, url):
        """
        Extract features from a URL for phishing detection.
        This should match the features used during training.
        """
        # Parse the URL
        parsed = urllib.parse.urlparse(url)

        # Example features - these should match what your training data expects
        return {
            'url_length': len(url),
            'hostname_length': len(parsed.hostname) if parsed.hostname else 0,
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscore': url.count('_'),
            'num_slash': url.count('/'),
            'num_questionmark': url.count('?'),
            'num_equal': url.count('='),
            'num_at': url.count('@'),
            'num_and': url.count('&'),
            # Add more features as needed based on your training data
        }


# Example usage:
if __name__ == "__main__":
    # To train a new model (uncomment if needed)
    detector = PhishingDetector()
    detector.train_model('phishing.csv', 'model/phishing_detection_model')

    # To use an existing model
    print(detector)
    detector = PhishingDetector('model/phishing_detection_model')

    # Example prediction
    test_url = "http://example.com/login.php?user=test&password=1234"
    result = detector.predict_url(test_url)
    print(f"URL: {result['url']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Phishing Probability: {result['phishing_probability']:.2f}")
    print(f"Legitimate Probability: {result['legitimate_probability']:.2f}")