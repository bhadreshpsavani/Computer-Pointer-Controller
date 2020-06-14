from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

class Head_Pose_Estimation_Model:
    '''
    This is a class for the operation of Head Pose Estimation Model
    '''
    
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        '''
        This will initiate Head Pose Estimation Model class object
        '''
        self.model_structure=model_name+'.xml'
        self.model_weights=model_name+'.bin'
        self.device_name=device
        self.threshold=threshold
        try:
            self.core=IECore()
            self.model=self.core(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        This method with load model using IECore object
        return loaded model
        '''
        try:
            self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        except Exception as e:
            print("Error While Loading the model",e) 

    def predict(self, image):
        '''
        This method will take image as a input and 
        does all the preprocessing, postprocessing
        '''
        p_image=self.preprocess_input(image)
        outputs=self.net.infer({self.input_name:p_image})
        outputs=outputs['detection_out']
        outputs=self.preprocess_output(outputs)
        return outputs

    def check_model(self):
        raise NotImplementedError

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
            print("Error While preprocessing Image",e) 
        return image

    def preprocess_output(self, outputs):
        '''
        Input: cordinates
        Description: Our model return cordinates in the form of array,
        Return: processed cordinates
        '''
        try:
            outputs=np.squeeze(outputs)
        except Exception as e:
            print("Error While preprocessing Outputs",e) 
        return outputs