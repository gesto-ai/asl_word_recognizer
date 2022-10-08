import json
import requests
from sign_recognizer.word_sign_recognizer import ASLWordRecognizer


class PredictorBackend:
    """Interface to a backend that serves predictions.
    To communicate with a backend accessible via a URL, provide the url kwarg.
    Otherwise, runs a predictor locally.
    """

    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = ASLWordRecognizer()
            self._predict = model.predict

    def run(self, video_url):
        pred = self._predict(video_url)
        return pred

    def _predict_from_endpoint(self, video_url):
        """Send an image to an endpoint that accepts JSON and return the predicted text.
        The endpoint should expect a video URL as a string
        under the key "video_url". It should return the predicted text under the key "prediction".
        Args:
            video_url: str
                A URL to a video to be converted into a string
        Returns
            pred: str
                The predictor's guess of the sign in the video.
        """

        headers = {"Content-type": "application/json"}
        payload = json.dumps({"video_url": video_url})
        response = requests.post(self.url, data=payload, headers=headers)
        pred = response.json()["prediction"]

        return pred
