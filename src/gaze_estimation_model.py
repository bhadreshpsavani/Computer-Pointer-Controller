from openvino.inference_engine import IECore, IENetwork
import cv2
import math

class Gaze_Estimation_Model:
    '''
    Class for the Gaze Estimation Model.
    '''
    
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        '''
        This will initiate Gaze Estimation Model class object
        '''
        self.model_structure=model_name+'.xml'
        self.model_weights=model_name+'.bin'
        print(self.model_structure, self.model_weights)
        self.device_name=device
        self.threshold=threshold
        try:
            self.core = IECore()
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            print("Error While Initilizing Gaze Estimation Model Class",e)
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        self.input_name=[i for i in self.model.inputs.keys()]
        print(self.input_name)
        self.input_shape=self.model.inputs[self.input_name[1]].shape
        self.output_name=[o for o in self.model.outputs.keys()]
        
    def load_model(self):
        '''
        This method with load model using IECore object
        return loaded model
        '''
        try:
            self.net = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
        except Exception as e:
            print("Error While Loading the model",e) 

    def predict(self, left_eye_image, right_eye_image, hpe_cords):
        '''
        This method will take image as a input and 
        does all the preprocessing, postprocessing
        '''
        left_eye_image=self.preprocess_img(left_eye_image)
        right_eye_image=self.preprocess_img(right_eye_image)
        outputs=self.net.infer({'left_eye_image':left_eye_image, 'right_eye_image':right_eye_image, 'head_pose_angles':hpe_cords})
        mouse_cord, gaze_vector=self.preprocess_output(outputs, hpe_cords)
        return mouse_cord, gaze_vector
        
    def preprocess_output(self, outputs, hpe_cords):
        '''
        Model output is dictionary like this
        {'gaze_vector': array([[ 0.51141196,  0.12343533, -0.80407059]], dtype=float32)}
        containing Cartesian coordinates of gaze direction vector
        
        We need to get this value and convert it in required format
        hpe_cords which is output of head pose estimation is in radian
        It needed to be converted in catesian cordinate 
        '''
        gaze_vector=outputs[self.output_name[0]][0]
        mouse_cord=(0,0)
        try:
            angle_r_fc=hpe_cords[2]
            sin_r=math.sin(angle_r_fc*math.pi/180.0)
            cos_r=math.cos(angle_r_fc*math.pi/180.0)
            x=gaze_vector[0]*cos_r+gaze_vector[1]*sin_r
            y=-gaze_vector[0]*sin_r+gaze_vector[1]*cos_r
            mouse_cord=(x,y)
        except Exception as e:
            print("Error While preprocessing output",e) 
        return mouse_cord, gaze_vector

    def preprocess_img(self, image):
        '''
        Input: image
        Description: We have done basic preprocessing steps for image
            1. Resizing image according to the model input shape
            2. Transpose of image to change the channels of image
            3. Reshape image
        Return: Preprocessed image
        '''
        try:
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose((2,0,1))
            image = image.reshape(1, *image.shape)
        except Exception as e:
            print("Error While preprocessing Image",e) 
        return image