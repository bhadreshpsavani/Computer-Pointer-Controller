from input_feeder import InputFeeder
from face_detection_model import Face_Detection_Model
from landmark_detection_model import Landmark_Detection_Model
import cv2

#load Models
face_detection_model=Face_Detection_Model('../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001')
face_detection_model.load_model()
landmark_detection_model=Landmark_Detection_Model('../intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009')
landmark_detection_model.load_model()

feeder=InputFeeder(input_type='video', input_file='../bin/demo.mp4')
feeder.load_data()
for batch in feeder.next_batch():
    detections, cropped_image=face_detection_model.predict(batch)
    cv2.imwrite('../imgs/cropped_image.jpg', cropped_image)
    left_eye_image, right_eye_image, eye_cords=landmark_detection_model.predict(cropped_image)
    print(left_eye_image.shape)
    print(right_eye_image.shape)
    print(eye_cords)
    cv2.imwrite('../imgs/left_eye.jpg', left_eye_image)
    cv2.imwrite('../imgs/right_eye.jpg', right_eye_image)
    break
feeder.close()