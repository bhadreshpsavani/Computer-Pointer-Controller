# Computer Pointer Controller

In this project, you will use a [Gaze Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) to control the mouse pointer of your computer. You will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project will demonstrate your ability to run multiple models in the same machine and coordinate the flow of data between those models.

You will be using the InferenceEngine API from Intel's OpenVino ToolKit to build the project. The gaze estimation model requires three inputs:

* The head pose
* The left eye image
* The right eye image.

![demoVideo](/bin/output_video.gif)

To get these inputs, you will have to use three other OpenVino models:

* [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

### The Pipeline:
You will have to coordinate the flow of data from the input, and then amongst the different models and finally to the mouse controller. The flow of data will look like this:

![pipeline](/imgs/pipeline.png)

## Project Set Up and Installation:

Step1: Download below three softwares:
1. Microsoft Visual Studio* with C++ 2019, 2017, or 2015 with MSBuild
2. CMake 3.4 or higher 64-bit
NOTE: If you want to use Microsoft Visual Studio 2019, you are required to install CMake 3.14.
3. Python 3.6.5 64-bit

Step2. Download **[OpenVino Toolkit 2020.1](https://docs.openvinotoolkit.org/latest/index.html)** with all the prerequisites by following this [installation guide](https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_windows.html)

Step3: Setup OpenVino Toolkit using below command in command prompt
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

Step4: Configure Model Optimizer using below commnads in command prompt
```
cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\install_prerequisites
install_prerequisites.bat
```

Step5: Varify installation
```
cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\demo\
demo_squeezenet_download_convert_run.bat
```
Above command should give output like this image
![optimizer_output](/imgs/image_classification_script_output_win.png)


## Demo:

Step1. Clone the Repository using `git clone https://github.com/bhadreshpsavani/Computer-Pointer-Controller.git`

Step2. Create Virtual Environment using command `python -m venv base` in the command prompt, then activate environment using below command,
```
cd base/Scripts/
activate
```

Step3. install all the dependency using `pip install requirements.txt`.

Step4. Instantiate OpenVino Environment. For windows use below command
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

Step5. Go back to the project directory `src` folder
```
cd path_of_project_directory
cd src
```

Step6. Run below commands to execute the project
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
- o : Specify path of output folder where we will store results
 
## Documentation: 

### Project Structure:

![project_structure](/imgs/project_structure.png)

intel: This folder contains models in IR format downloaded from Openvino Model Zoo

src: This folder contains model files, pipeline file(main.py) and utilities 
* `model.py` is the model class file which has common property of all the other model files. It is inherited by all the other model files 
This folder has 4 model class files, This class files has methods to load model and perform inference.
* `face_detection_model.py`
* `gaze_estimation_model.py`
* `landmark_detection_model.py`
* `head_pose_estimation_model.py`
* `main.py` file used to run complete pipeline of project. It calls has object of all the other class files in the folder
* `mouse_controller.py` is utility to move mouse curser based on mouse coordinates received from  `gaze_estimation_model` class predict method.
* `input_feeder.py` is utility to load local video or webcam feed
* `banchmark.ipynb`, `computer_controller_job.sh`, and `PrecisionComparsion.ipynb` are for banchmarking result generation for different hardware and model comparison

bin: this folder has `demo.mp4` file which can be used to test model

## Benchmarks
I have checked `Inference Time`, `Model Loading Time`, and `Frames Per Second` model for `FP16`, `FP32`, and `FP32-INT8` of all the models except `Face Detection Model`. `Face Detection Model` was only available on `FP32-INT1` precision. 
You can use below commands to get results for respective precisions,

`FP16`: 
```
python main.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \ 
-hp ../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \ 
-ge ../intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml \ 
-d CPU -i ../bin/demo.mp4 -o results/FP16/ -flags ff fl fh fg
```

`FP32`: 
```
python main.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr ../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \ 
-hp ../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml \ 
-ge ../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml \ 
-d CPU -i ../bin/demo.mp4 -o results/FP32/ -flags ff fl fh fg
```

`FP32-INT8`: 
```
python main.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr ../intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009.xml \ 
-hp ../intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml \ 
-ge ../intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml \ 
-d CPU -i ../bin/demo.mp4 -o results/FP32-INT8/ -flags ff fl fh fg
```

### Inference Time:
<img src="/imgs/inference_time.png" width="425"/> <img src="/imgs/inference_time_a.png" width="425"/> 

### Model Loading Time:
<img src="/imgs/model_loading_time.png" width="425"/> <img src="/imgs/model_loading_time_a.png" width="425"/> 

### Frames Per Second:
<img src="/imgs/fps.png" width="425"/> <img src="/imgs/fps_a.png" width="425"/>

**Synchronous Inference**

```
precisions = ['FP16', 'FP32', 'FP32-INT8']
Inference Time : [26.6, 26.4, 26.9]
fps : [2.218045112781955, 2.234848484848485, 2.193308550185874]
Model Load Time : [1.6771371364593506, 1.6517729759216309, 5.205628395080566]
```

**Asynchronous Inference**

```
precisions = ['FP16', 'FP32', 'FP32-INT8']
Inference Time : [23.9, 24.7, 24.0]
fps : [2.468619246861925, 2.388663967611336, 2.4583333333333335]
Model Load Time : [0.7770581245422363, 0.7230548858642578, 2.766681432723999]
```

## Results:
* From above observations we can say that `FP16` has lowest model time and `FP32-INT8` has highest model loading time, the reason for the higher loading time can be said as combination of precisions lead to higher weight of the model for `FP32-INT8`.
* For `Inference Time` and `FPS`, `FP32` give slightly better results. There is not much difference for this three different models for this two parameters.
* I have tested model for Asynchronous Inference and Synchronous Inference, Asynchronous Inference has better results it has slight improvement in `inference time` and `FPS`

### Edge Cases
* Multiple People Scenario: If we encounter multiple people in the video frame, it will always use and give results one face even though multiple people detected,  
* No Head Detection: it will skip the frame and inform the user

### Area of Improvement:
* lighting condition: We might use HSV based pre-processing steps to minimize error due to different lighting conditions 
