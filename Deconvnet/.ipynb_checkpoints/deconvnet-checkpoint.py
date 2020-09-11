import numpy as np
from keras.layers import Input, InputLayer, Flatten, Activation, Dense, Conv2D, MaxPool2D
from Deconvnet.d_layers import DConvolution2D, DPooling, DInput, DActivation



class Deconvnet():
    def __init__(self, model, num_layers):
        self.model = model
        self.num_layers = num_layers
        self.deconv_layers = []
        self.set_layers()
        
    def forward(self, data):
        self.deconv_layers[0].up(data)
        for i in range(1, len(self.deconv_layers)):
            self.deconv_layers[i].up(self.deconv_layers[i - 1].up_data)

        output = self.deconv_layers[-1].up_data
        return output
    
    def backwards(self, data, ix=None):
        
        if ix is not None:
            feature_map = data[:, :, :, ix]
            data = np.zeros_like(data)
            data[:, :, :, ix] = feature_map
            
        
        self.deconv_layers[-1].down(data)

        for i in range(len(self.deconv_layers) - 2, -1, -1):
            self.deconv_layers[i].down(self.deconv_layers[i + 1].down_data)

        deconv = self.deconv_layers[0].down_data
        deconv = deconv.squeeze()
        return deconv
        
        
    def set_layers(self):
        for i in range(self.num_layers):
            if isinstance(self.model.layers[i], Conv2D):
                self.deconv_layers.append(DConvolution2D(self.model.layers[i]))
                self.deconv_layers.append(
                        DActivation(self.model.layers[i]))
            elif isinstance(self.model.layers[i], MaxPool2D):
                self.deconv_layers.append(DPooling(self.model.layers[i]))
            elif isinstance(self.model.layers[i], InputLayer):
                self.deconv_layers.append(DInput(self.model.layers[i]))
            else:
                print('Cannot handle this type of layer')
                break