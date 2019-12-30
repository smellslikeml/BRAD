#!/usr/bin/env python3
import cv2
import numpy as np
from time import sleep, time
from collections import deque
from ast import literal_eval

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.callbacks import History


def build_model(img_size, net_arch):

    input_img = Input(shape=(img_size, img_size, 1))

    # Adding Encoder Layers
    for idx, layer in enumerate(net_arch):
        kernel = (layer[1], layer[1])
        pooling = (layer[2], layer[2])
        if not idx:
            # First layer accepts keras Input
            x = Conv2D(layer[0], kernel, activation='relu', padding='same')(input_img)
        else:
            x = Conv2D(layer[0], kernel, activation='relu', padding='same')(x)
        x = MaxPooling2D(pooling, padding='same')(x)

    # Final Encoder Layer
    layer = net_arch[-1]
    kernel = (layer[1], layer[1])
    pooling = (layer[2], layer[2])
    x = Conv2D(layer[0], kernel, activation='relu', padding='same')(x)
    x = UpSampling2D(pooling)(x)

    # Adding Decoder Layers
    for idx, layer in enumerate(net_arch[::-1][1:]):
        kernel = (net_arch[idx - 1][1], net_arch[idx - 1][1])
        pooling = (net_arch[idx - 1][2], net_arch[idx - 1][2])
        x = Conv2D(layer[0], kernel, activation='relu', padding='same')(x)
        x = UpSampling2D(pooling)(x)

    # Final Decoder Layer
    layer = net_arch[0]
    kernel = (layer[1], layer[1])
    x = Conv2D(1, kernel, activation='sigmoid', padding='same')(x)

    # Build Model from Layers
    model = Model(input_img, x)

    # Compile and print summary
    model.compile(optimizer=RMSprop(lr=float(config['model']['learning_rate'])), loss='binary_crossentropy')
    print(model.summary())
    return model

def load_run_render(model, frame):
    # Resize & Grayscale input images, reshape array
    im = cv2.resize(frame, (img_size, img_size))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = np.expand_dims(np.expand_dims(im, axis=0).astype(np.float32), axis=-1) / 255

    # Fit model and return loss
    history = History()
    model.fit(im, im,
                epochs=1,
                batch_size=1,
                shuffle=False,
                callbacks=[history], 
                verbose=0)
    return history.history['loss'][-1]

def main():
    while True:
        for idx, source in enumerate(source_dict):
            try:
                ret, frame = source_dict[source].read()
                ts = int(time() * 1000)
                if ret:
                    loss = load_run_render(model_dict[idx], frame)
                    q_dict[idx].append(loss)
                    if len(q_dict[idx]) > int(config['model']['initial_steps']):
                        prev_hist = list(q_dict[idx])[:-1]
                        threshold = float(config['threshold']['sigma']) * np.std(prev_hist)
                        if np.abs(loss - np.mean(prev_hist)) > threshold:
                            print(idx, loss, ' ============ anomaly')
                        else:
                            print(idx, loss, 'normal')
                    else:
                        pass
            except Exception as e:
                print(e)
                pass
            sleep(float(config['delay']['seconds']))
        

if __name__ == '__main__':
    import configparser

    config = configparser.ConfigParser()
    config.read('config.ini')

    sourceList = (config['sources']['sourceList']).split(',')
    img_size = int(config['img']['size'])
    net_arch = literal_eval(config['model']['net_arch'])

    q_dict = {}
    model_dict = {}
    source_dict = {}

    for idx, source in enumerate(sourceList):
        try:
            source_dict[idx] = cv2.VideoCapture(source)
            model_dict[idx] = build_model(img_size=img_size, net_arch=net_arch)
            q_dict[idx] = deque(maxlen=int(config['queue']['length']))
        except:
            break

    main()
