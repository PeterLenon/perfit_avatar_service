import os
from fastapi import FastAPI
import cv2
import mediapipe as mp
import yaml
from types import SimpleNamespace
from redis import Redis, ResponseError
from loguru import logger

app = FastAPI()

def _load_config():
    configuration_path = os.getenv("AVATAR_SERVICE_CONFIG", "config.yaml")
    with open(configuration_path, "r") as file:
        config = yaml.safe_load(file)
    return SimpleNamespace(**config)

configuration_params = _load_config()
redis_client = Redis(
    host=configuration_params.REDIS_PARAMETERS.HOST,
    port=configuration_params.REDIS_PARAMETERS.PORT,
    decode_responses=True,
)

def _valid_user_image(user_img):
    if user_img is None:
        return False
    height, width = user_img.shape[:2]
    if height < configuration_params.IMAGE_PARAMETERS.MIN_HEIGHT or width < configuration_params.IMAGE_PARAMETERS.MIN_WIDTH:
        return False

    grayscale = cv2.cvtColor(user_img, cv2.COLOR_BGR2GRAY)
    image_brightness = grayscale.mean()
    if image_brightness < configuration_params.IMAGE_PARAMETERS.MIN_BRIGHTNESS:
        return False

    face_landmarks = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    face_detection_results = face_landmarks.process(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))
    if not face_detection_results.multi_face_landmarks:
        return False
    return True

"""
this create function will invoke an avatar creation workflow and return the workflow id to check if workflow is 
completed/failed/restarted. The client code calling this end point will check for progress of this workflow to see if 
it can now poll results from S3 storage for the completed avatar for this user
"""
@app.post("/create")
def create_user_avatar(payload: dict):
    payload = dict(payload)
    user_img = payload.get("img")
    if not _valid_user_image(user_img):
        return {
            "error": "Invalid image"
        }
    message_id = redis_client.xadd(
        configuration_params.REDIS_PARAMETERS.STREAM_NAME,
        payload
    )
    return {"message_id": message_id}







