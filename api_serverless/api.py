import json
from sign_recognizer.word_sign_recognizer import ASLWordRecognizer, process_video, convert_y_label_to_string

model = ASLWordRecognizer()

def handler(event, _context):
    print("INFO loading video from URL")
    video = _load_video(event)
    if video is None:
        return {"statusCode": 400, "message": "`video_url` not found in event"}
    print("INFO video loaded")
    
    print("INFO starting inference")
    pred = model.predict_on_video(video)
    pred_str = convert_y_label_to_string(y=pred[-1], mapping=model.mapping)
    print("INFO inference complete")
    print("INFO pred {}".format(pred_str))

    return {"prediction": str(pred_str)}

def _load_video(event):
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    video_url = event.get("video_url")
    if video_url is not None:
        print("INFO url {}".format(video_url))
        print("INFO Note: We're hard-coding the number of frames to be 74 for now, until we find a better way to select the right frames from any given video!")
        return process_video(video_url, 1, 74)
    return None


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event
