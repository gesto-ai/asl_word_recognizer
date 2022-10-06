import json
from sign_recognizer.word_sign_recognizer import ASLWordRecognizer
from sign_recognizer.word_sign_recognizer import process_video

model = ASLWordRecognizer()

def handler(event, _context):
    print("INFO loading video")
    print(event)
    video = _load_video(event)
    print(f"What is video? {video}")
    
    if video is None:
        return {"statusCode": 400, "message": "neither video_url nor image found in event"}
    print("INFO video loaded")
    print("INFO starting inference")
    
    # pred = model.predict(image)
    # print("INFO inference complete")
    # image_stat = ImageStat.Stat(image)
    # print("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
    # print("METRIC image_area {}".format(image.size[0] * image.size[1]))
    # print("METRIC pred_length {}".format(len(pred)))
    # print("INFO pred {}".format(pred))
    # return {"pred": str(pred)}
    return {"pred": "Hello world!"}

def _load_video(event):
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    video_url = event.get("video_url")
    if video_url is not None:
        print("INFO url {}".format(video_url))
        return process_video(video_url, 1, 74)
        # return util.read_image_pil(video_url, grayscale=True)
    else:
        video = event.get("image")
        if video is not None:
            print("INFO reading image from event")
            return "LMAO" #util.read_b64_image(image, grayscale=True)
        else:
            return None


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event
