from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2
import logging

class Landmark_Detection_Model:
    '''
    This is a class for the operation of Landmark Detection Model
    '''
    
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        '''
        This will initiate Landmark Detection Model class object
        '''
        self.logger=logging.getLogger('ld')
        self.model_structure=model_name
        self.model_weights=model_name.replace('.xml','.bin')
        self.device_name=device
        self.threshold=threshold
        try:
            self.core = IECore()
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            self.logger.error("Error While Initilizing Landmark Detection Model Class",e)
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))

    def load_model(self):
        '''
        This method with load model using IECore object
        return loaded model
        '''
        try:
            self.net = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
        except Exception as e:
            self.logger.error("Error While Loading Landmark Detection Model"+str(e)) 

    def predict(self, image):
        '''
        This method will take image as a input and 
        does all the preprocessing, postprocessing
        '''
        left_eye_image, right_eye_image, eye_cords=[], [], []
        try:
            p_image=self.preprocess_input(image)
            outputs=self.net.infer({self.input_name:p_image})
            left_eye_image, right_eye_image, eye_cords=self.preprocess_output(outputs, image)
        except Exception as e:
            self.logger.error("Error While making prediction in Landmark Detection Model"+str(e)) 
        return left_eye_image, right_eye_image, eye_cords
        
    def preprocess_output(self, outputs, image):
        '''
        The net outputs a blob with the shape: [1, 10], 
        containing a row-vector of 10 floating point values 
        for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5). 
        All the coordinates are normalized to be in range [0,1].
        We only need 
        '''
        h=image.shape[0]
        w=image.shape[1]
        left_eye_image, right_eye_image, eye_cords=[], [], []
        try:
            outputs=outputs[self.output_name][0]
            
            left_eye_xmin=int(outputs[0][0][0]*w)-10
            left_eye_ymin=int(outputs[1][0][0]*h)-10
            right_eye_xmin=int(outputs[2][0][0]*w)-10
            right_eye_ymin=int(outputs[3][0][0]*h)-10
            
            left_eye_xmax=int(outputs[0][0][0]*w)+10
            left_eye_ymax=int(outputs[1][0][0]*h)+10
            right_eye_xmax=int(outputs[2][0][0]*w)+10
            right_eye_ymax=int(outputs[3][0][0]*h)+10
            
            left_eye_image=image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
            right_eye_image=image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]
            
            eye_cords=[[left_eye_xmin, left_eye_xmax, left_eye_xmax, left_eye_ymax],[right_eye_xmin, right_eye_xmax, right_eye_ymin, right_eye_ymax]]
            
        except Exception as e:
            self.logger.error("Error While drawing bounding boxes on image in Landmark Detection Model"+str(e)) 
        return left_eye_image, right_eye_image, eye_cords

    def preprocess_input(self, image):
        '''
        Input: image
        Description: We have done basic preprocessing steps
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
            self.logger.error("Error While preprocessing Image in Landmark Detection Model"+str(e)) 
        return image