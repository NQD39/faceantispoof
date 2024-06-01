# anti_spoof_predict.py
import onnxruntime
import numpy as np
import cv2

class AntiSpoofPredict:
    def __init__(self, device_id, model_path):
        self.device_id = device_id
        self.model_path = model_path
        self.session = onnxruntime.InferenceSession(model_path, providers=[ "CPUExecutionProvider"])

    def get_bbox(self, image):
        # Dummy implementation of get_bbox, replace with actual method
        height, width, _ = image.shape
        return [0, 0, width, height]

    def predict(self, img):
        img = cv2.resize(img, (80, 80))  # Ensure the input size matches the model's expected input size
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.transpose(img, (0, 3, 1, 2))  # Change to NCHW format

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        result = self.session.run([output_name], {input_name: img})[0]
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        # Apply softmax to the output to get probabilities
        softmax_output = softmax(result[0])
        return softmax_output
