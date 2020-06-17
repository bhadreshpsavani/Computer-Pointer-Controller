import cv2
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection_model import Face_Detection_Model
from landmark_detection_model import Landmark_Detection_Model
from head_pose_estimation_model import Head_Pose_Estimation_Model
from gaze_estimation_model import Gaze_Estimation_Model

def main():
    
    #load Models
    face_detection_model=Face_Detection_Model('../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001')
    landmark_detection_model=Landmark_Detection_Model('../intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009')
    head_pose_estimation_model=Head_Pose_Estimation_Model('../intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001')
    gaze_estimation_model=Gaze_Estimation_Model('../intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002')     

    
    face_detection_model.load_model()
    landmark_detection_model.load_model()
    head_pose_estimation_model.load_model()
    gaze_estimation_model.load_model()

    
    feeder=InputFeeder(input_type='video', input_file='../bin/demo.mp4')
    feeder.load_data()
    
    mouse_controller=MouseController('medium','fast')
    
    frame_count=0
    for frame in feeder.next_batch():
        
        frame_count+=1
        print("frame_count", frame_count)
        if frame_count%5==0:
            cv2.imshow('video', cv2.resize(frame, (500, 500)))
            
        detections, cropped_image=face_detection_model.predict(frame)
        left_eye_image, right_eye_image, eye_cords=landmark_detection_model.predict(cropped_image)
        pose_output=head_pose_estimation_model.predict(cropped_image)
        mouse_cord, gaze_vector=gaze_estimation_model.predict(left_eye_image, right_eye_image, pose_output)
        
        if frame_count%5==0:
            mouse_controller.move(mouse_cord[0], mouse_cord[1])
            
    cv2.destroyAllWindows()
    feeder.close()
    
if __name__=='__main__':
    main()
