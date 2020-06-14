from src.input_feeder import InputFeeder
from src.face_detection_model import Face_Detection_Model

#load Models
face_detection_model=Face_Detection_Model('intel/face-detection-adas-binary-0001/INT32-INT1/face-detection-adas-binary-0001')
face_detection_model.load_model()

feeder=InputFeeder(input_type='video', input_file='bin/demo.mp4')
feeder.load_data()
for batch in feeder.next_batch():
    image=batch
    detections, p_image=face_detection_model.predict(image)
    print(detections)
    print("image is :")
    print(image)
    break
feeder.close()