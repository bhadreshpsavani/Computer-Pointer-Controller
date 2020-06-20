# Computer Pointer Controller

In this project, you will use a [Gaze Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) to control the mouse pointer of your computer. You will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project will demonstrate your ability to run multiple models in the same machine and coordinate the flow of data between those models.

You will be using the InferenceEngine API from Intel's OpenVino ToolKit to build the project. The gaze estimation model requires three inputs:

* The head pose
* The left eye image
* The right eye image.

To get these inputs, you will have to use three other OpenVino models:

* [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

### The Pipeline:
You will have to coordinate the flow of data from the input, and then amongst the different models and finally to the mouse controller. The flow of data will look like this:

![pipeline](/imgs/pipeline.png)

## Project Set Up and Installation:

Step1. Download **[OpenVino Toolkit 2020.1](https://docs.openvinotoolkit.org/latest/index.html)** with all the prerequisites by following this [installation guide](https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_windows.html)

Step2. Clone the Repository using `git clone https://github.com/denilDG/Computer-Pointer-Controller.git`

Step3. Create Virtual Environment using command `virtualenv venv` in the command prompt

Step4. install all the dependency using `pip install requirements.txt`

## Demo:

Step1. Open command prompt Activate Virtual Environment 
```
cd venv/Scripts/
activate
```

Step2. Instantiate OpenVino Environment. For windows use below command
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

Step3. Go back to the project directory `src` folder
```
cd path_of_project_directory
cd src
```

Step4. Run below commands to execute the project
```
python main.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr ../intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009.xml \ 
-hp ../intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml \ 
-ge ../intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml \ 
-i ../bin/demo.mp4 -flags ff fl fh fg
```
Command Line Argument Information:
- fd : Specify path of xml file of face detection model
- lr : Specify path of xml file of landmark regression model
- hp : Specify path of xml file of Head Pose Estimation model
- ge : Specify path of xml file of Gaze Estimation model
- i : Specify path of input Video file or cam for Webcam
- flags (Optional): if you want to see preview video in separate window you need to Specify flag from ff, fl, fh, fg like -flags ff fl...(Space seperated if multiple values) ff for faceDetectionModel, fl for landmarkRegressionModel, fh for headPoseEstimationModel, fg for gazeEstimationModel
- probs (Optional): if you want to specify confidence threshold for face detection, you can specify the value here in range(0, 1),  default=0.6
- d (Optional): Specify Device for inference, the device can be CPU, GPU, FPGU, MYRID
 
## Documentation: 

### Project Structure:

![project_structure](/imgs/project_structure.png)

intel: This folder contains models in IR format downloaded from Openvino Model Zoo

src: This folder has 4 model class files, This class files has methods to load model and perform inference.
* `face_detection_model.py`
* `gaze_estimation_model.py`
* `landmark_detection_model.py`
* `head_pose_estimation_model.py`
* `main.py` file used to run complete pipeline of project. It calls has object of all the other class files in the folder
* `mouse_controller.py` is utility to move mouse curser based on mouse coordinates received from  `gaze_estimation_model` class predict method.
* `input_feeder.py` is utility to load local video or webcam feed

bin: this folder has `demo.mp4` file which can be used to test model

### 

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
