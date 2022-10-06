from sign_recognizer.word_sign_recognizer import ASLWordRecognizer
import sys

def handler(event, context):
    return 'Hello from AWS Lambda using Python' + sys.version + '!'  