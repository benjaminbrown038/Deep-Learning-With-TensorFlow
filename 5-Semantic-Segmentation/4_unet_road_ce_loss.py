
def unet(num_classes, shape):
    
    model_input = Input(shape=shape)
        
    # Encoder_block-1.
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(model_input)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # Encoder_block-2.
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Encoder_block-3.
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Encoder_block-4.
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Intermedicate_block.
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder_block-1.
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2),padding = 'same')(c5)
    # Lateral connection from Encoder_block-4.
    u6 = concatenate([u6,c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding= 'same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu',  padding= 'same')(c6)

    # Decoder_block-2.
    u7 = Conv2DTranspose(256, (2,2), strides = (2, 2), padding= 'same')(c6)
    # Lateral connection from Encoder_block-3.
    u7 = concatenate([u7,c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding= 'same')(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu',  padding= 'same')(c7)

    # Decoder_block-3.
    u8 = Conv2DTranspose(128, (2,2), strides= (2, 2),padding = 'same')(c7)
    # Lateral connection from Encoder_blcok-2.
    u8 = concatenate([u8,c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding= 'same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu',   padding= 'same')(c8)

    # Decoder_block-4.
    u9 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding= 'same')(c8)
    # Lateral connection from Encoder_blcok-1.
    u9 = concatenate([u9,c1], axis =3)
    c9 = Conv2D(64, (3, 3), activation ='relu',  padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation ='relu',  padding='same')(c9)

    # 1x1 convolution to limit the depth of the feature maps to the number of classes.
    outputs = Conv2D(num_classes, (1, 1), use_bias=False)(c9)

    model_output = Activation('softmax')(outputs)

    model = Model(inputs=model_input, outputs=model_output)

    return model
