from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2
import logging

class Face_Detection_Model:
    '''
    This is a class for the operation of Face Detection Model
    '''
    
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        '''
        This will initiate Face Detection Model class object
        '''
        self.model_structure=model_name
        self.model_weights=model_name.replace('.xml','.bin')
        self.device_name=device
        self.threshold=threshold
        self.logger=logging.getLogger('fd')
        try:
            self.core = IECore()
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            self.logger.error("Error While Initilizing Face Detection Model Class"+str(e))
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
            self.net = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
        except Exception as e:
            self.logger.error("Error While Loading the model",e) 

    def predict(self, image):
        '''
        This method will take image as a input and 
        does all the preprocessing, postprocessing
        '''
        try:
            p_image=self.preprocess_input(image)
            outputs=self.net.infer({self.input_name:p_image})
            outputs=outputs['detection_out']
            detections, cropped_image=self.preprocess_output(outputs, image)
        except Exception as e:
            self.logger.error("Error While Prediction in Face Detection Model"+str(e))
        return detections, cropped_image

    def check_model(self):
        raise NotImplementedError
        
    def preprocess_output(self, coords, image):
        '''
        We will have multiple detection for single image
        This function will take image and preprocessed cordinates
        and return image with bounding boxes and scaled cordinates
        '''
        width, height = int(image.shape[1]), int(image.shape[0])
        detections=[]
        cropped_image=image
        coords=np.squeeze(coords)
        try:
            for coord in coords:
                image_id, label, threshold, xmin, ymin, xmax, ymax=coord
                if image_id==-1:
                    break
                if label==1 and threshold>=self.threshold:
                    xmin=int(xmin*width)
                    ymin=int(ymin*height)
                    xmax=int(xmax*width)
                    ymax=int(ymax*height)
                    detections.append((xmin,ymin, xmax, ymax))
                    cropped_image=image[ymin:ymax, xmin:xmax]
        except Exception as e:
            self.logger.error("Error While drawing bounding boxes on image in Face Detection Model"+str(e))
        return detections, cropped_image

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
            self.logger.error("Error While preprocessing Image in Face Detection Model"+str(e))
        return image