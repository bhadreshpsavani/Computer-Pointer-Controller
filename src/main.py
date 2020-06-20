import cv2
import os
import logging
import time
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection_model import Face_Detection_Model
from landmark_detection_model import Landmark_Detection_Model
from head_pose_estimation_model import Head_Pose_Estimation_Model
from gaze_estimation_model import Gaze_Estimation_Model
from argparse import ArgumentParser

def build_argparser():
    """
    parse commandline argument
    return ArgumentParser object
    """
    parser=ArgumentParser()
    parser.add_argument("-fd", "--faceDetectionModel", type=str,
                       help="Specify path of xml file of face detection model",
                       default="../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml")
    
    parser.add_argument("-lr", "--landmarkRegressionModel", type=str,
                       help="Specify path of xml file of landmark regression model",
                        default="../intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009.xml")
    
    parser.add_argument("-hp", "--headPoseEstimationModel", type=str,
                       help="Specify path of xml file of Head Pose Estimation model",
                        default="../intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml")
    
    parser.add_argument("-ge", "--gazeEstimationModel", type=str,
                       help="Specify path of xml file of Gaze Estimation model",
                        default="../intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml")
    
    parser.add_argument("-i", "--input", type=str, default='../bin/demo.mp4',
                       help="Specify path of input Video file or cam for Webcam")
    
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                       help="Specify flag from ff, fl, fh, fg like -flags ff fl(Space seperated if multiple values)"
                       "ff for faceDetectionModel, fl for landmarkRegressionModel"
                       "fh for headPoseEstimationModel, fg for gazeEstimationModel")
    
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                       help="Specify probability threshold for face detection model")
    
    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                       help="Specify Device for inference"
                       "It can be CPU, GPU, FPGU, MYRID")
    parser.add_argument("-o", '--output_path', default='/results', type=str)
    return parser


def main():
    
    args=build_argparser().parse_args()
    logger=logging.getLogger('main')
    
    is_benchmarking=False
    #initialize variables with the input arguments for easy access
    modelPathDict={
        'FaceDetectionModel':args.faceDetectionModel,
        'LandmarkRegressionModel':args.landmarkRegressionModel,
        'HeadPoseEstimationModel':args.headPoseEstimationModel,
        'GazeEstimationModel':args.gazeEstimationModel
    }
    previewFlags=args.previewFlags
    input_filepath=args.input
    device_name=args.device
    prob_threshold=args.prob_threshold
    output_path = args.output_path

    if input_filepath.lower()=='cam':
        feeder=InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_filepath):
            logger.error("Unable to find specified video file")
            exit(1)
        feeder=InputFeeder(input_type='video', input_file=input_filepath)
    
    for model_path in list(modelPathDict.values()):
        if not os.path.isfile(model_path):
            logger.error("Unable to find specified model file"+str(model_path))
            exit(1)
            
    # instantiate model
    face_detection_model=Face_Detection_Model(modelPathDict['FaceDetectionModel'], device_name)
    landmark_detection_model=Landmark_Detection_Model(modelPathDict['LandmarkRegressionModel'], device_name)
    head_pose_estimation_model=Head_Pose_Estimation_Model(modelPathDict['HeadPoseEstimationModel'], device_name)
    gaze_estimation_model=Gaze_Estimation_Model(modelPathDict['GazeEstimationModel'], device_name)     
    
    if not is_benchmarking:
        mouse_controller=MouseController('medium','fast')

    # load Models
    start_model_load_time = time.time()
    face_detection_model.load_model()
    landmark_detection_model.load_model()
    head_pose_estimation_model.load_model()
    gaze_estimation_model.load_model()
    total_model_load_time = time.time() - start_model_load_time

    feeder.load_data()

    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), feeder.get_fps(),
                                (1920, 1080), True)
    
    frame_count=0
    start_inference_time = time.time()
    for ret, frame in feeder.next_batch():
        
        if not ret:
            break

        frame_count+=1
        logger.error("frame_count"+str(frame_count))
        
        if frame_count%5==0:
            cv2.imshow('video', cv2.resize(frame, (500, 500)))
            
        key=cv2.waitKey(60)

        try:
            face_cords, cropped_image=face_detection_model.predict(frame)

            if type(cropped_image)==int:
                logger.error("Unable to detect the face")
                if key==27:
                    break
                continue

            left_eye_image, right_eye_image, eye_cords=landmark_detection_model.predict(cropped_image)
            pose_output=head_pose_estimation_model.predict(cropped_image)
            mouse_cord, gaze_vector=gaze_estimation_model.predict(left_eye_image, right_eye_image, pose_output)
        except Exception as e:
            logger.error("Could predict using model"+str(e))
            continue

        
        if not len(previewFlags)==0 and not is_benchmarking:
            previewFrame=frame.copy()
            if 'ff' in previewFlags:
                cv2.rectangle(previewFrame, (face_cords[0][0], face_cords[0][1]), (face_cords[0][2], face_cords[0][3]), (0,255,0), 3)
                previewFrame[face_cords[0][1]:face_cords[0][3], face_cords[0][0]:face_cords[0][2]]=cropped_image
            if 'fl' in previewFlags:
                cv2.rectangle(cropped_image, (eye_cords[0][0], eye_cords[0][1]), (eye_cords[0][1], eye_cords[0][3]), (0,255,0), 3)
                cv2.rectangle(cropped_image, (eye_cords[1][0], eye_cords[1][1]), (eye_cords[1][1], eye_cords[1][3]), (0,255,0), 3)
                previewFrame[face_cords[0][1]:face_cords[0][3], face_cords[0][0]:face_cords[0][2]]=cropped_image
            if 'fh' in previewFlags:
                cv2.putText(
                    previewFrame, 
                    "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(pose_output[0], pose_output[1], pose_output[2]), 
                    (20, 40),
                    cv2.FONT_HERSHEY_COMPLEX, 
                    1, (255, 0, 255), 2)
            if 'fg' in previewFlags:
                x, y, w=int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le=cv2.line(left_eye_image.copy(), (x-w, y-w), (x+w, y+w), (255, 0, 255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255, 0, 255), 2)
                re=cv2.line(right_eye_image.copy(), (x-w, y-w), (x+w, y+w), (255, 0, 255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255, 0, 255), 2)
                cropped_image[eye_cords[0][1]:eye_cords[0][3], eye_cords[0][0]:eye_cords[0][2]] = le
                cropped_image[eye_cords[1][1]:eye_cords[1][3], eye_cords[1][0]:eye_cords[1][2]] = re
                previewFrame[face_cords[0][1]:face_cords[0][3], face_cords[0][0]:face_cords[0][2]]=cropped_image
            cv2.imshow('preview', cv2.resize(previewFrame, (500, 500)))
            out_video.write(previewFrame)
        
        logger.error("Mouse Cordinates:"+str(mouse_cord))
        
        if frame_count%5==0 and not is_benchmarking:
            mouse_controller.move(mouse_cord[0], mouse_cord[1])
        
        if key==27:
            break

    total_time = time.time() - start_inference_time
    total_inference_time = round(total_time, 1)
    fps = frame_count / total_inference_time
    
    if is_benchmarking:
        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time) + '\n')
            f.write(str(fps) + '\n')
            f.write(str(total_model_load_time) + '\n')

    logger.error('Model load time: '+str(total_model_load_time))
    logger.error('Inference time: '+str(total_inference_time))
    logger.error('FPS: '+str(fps))

    logger.error("Videostream ended")     
    cv2.destroyAllWindows()
    feeder.close()
    
if __name__=='__main__':
    main()
