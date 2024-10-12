       

def fcn32s_vgg16(num_classes,shape):
  
    model_input = Input(shape=shape)

    # Conv Block 1 
    x = Conv2D(64)
    x = Activation()
    x = Conv2D(64)
    x = Activation()
    x = MaxPool2D()

    # Conv Block 2 
    x = Conv2D(128)
    x = Activation()
    x = Conv2D(128)
    x = Activation()
    x = MaxPool2D()

    # Conv Block 3 
    x = Conv2D(256)
    x = Activation()
    x = Conv2D(256)
    x = Activation()
    x = Conv2D(256)
    x = Activation()
    x = MaxPool2D()

    # Conv Block 4 
    x = Conv2D()
    x = Activation()
    x = Conv2D()
    x = Activation()
    x = Conv2D()
    x = Activation()
    x = MaxPool2D()

    x = Conv2D()

    outputs = Conv2DTranspose()

    model_output = Activation()


    
