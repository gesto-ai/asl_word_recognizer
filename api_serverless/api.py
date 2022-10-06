import json
from sign_recognizer.word_sign_recognizer import ASLWordRecognizer
from sign_recognizer.word_sign_recognizer import process_video

model = ASLWordRecognizer()

def handler(event, _context):
    print("INFO loading video")
    video = _load_video(event)

    if video is None:
        return {"statusCode": 400, "message": "neither video_url nor image found in event"}
    print("INFO video loaded")
    print("INFO starting inference")
    pred = model.predict_on_video(video)
    print("INFO inference complete")
    print("INFO pred {}".format(pred))
    return {"pred": str(pred)}

def _load_video(event):
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    video_url = event.get("video_url")
    if video_url is not None:
        print("INFO url {}".format(video_url))
        return process_video(video_url, 1, 74)
        # return util.read_image_pil(video_url, grayscale=True)
    else:
        video = event.get("video")
        if video is not None:
            print("INFO reading video from event")
            return "LMAO" #util.read_b64_image(image, grayscale=True)
        else:
            return None


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event



# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"msg": "hello"}'
# curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"video_url": "https://drive.google.com/uc?export=download&id=1lWdgnNbkosDJ_7p7_qwyBuKqCYs1yvEI"}'