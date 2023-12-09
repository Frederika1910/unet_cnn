import cv2
import json
import numpy as np
from efficientnet.model import models, EfficientNetB0
from keras import Model
from keras.src.layers import Flatten, Dense
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Input
from efficientunet import *
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_lymphocyte_coordinates(image_info, annotation_info):
    # Load image
    path = 'C:/Users/frede/Desktop/FRI/Ing/3. semester/Projekt/UNet/data'
    # path = 'C:/Users/frede/Desktop/FRI/Ing/2. semester/Projekt/TIGER/wsirois/roi-level-annotations/tissue-cells/images'
    image_path = path + image_info['file_name'][8:]
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    lymphocyte_coordinates = []
    if image is None:
        return image, lymphocyte_coordinates

    bbox = annotation_info['bbox']
    if (len(bbox) > 0):
        x, y, w, h = map(int, bbox)
        lymphocyte_coordinates.append((x, y))
    else:
        print('neni su')

    return image, lymphocyte_coordinates


if __name__ == '__main__':
    json_file = 'tmp.json'
    data = load_json(json_file)

    images_info = {image['id']: image for image in data['images']}
    annotations_info = data['annotations']

    # input_shape = (256, 256, 3)
    # base_model = keras.applications.EfficientNetB0(
    #     include_top=False,
    #     weights='imagenet',
    #     input_shape=(256, 256, 3),
    # )
    #
    # x = base_model.output
    # x = Flatten()(x)
    # x = Dense(1024, activation='relu')(x)
    # output_layer = Dense(2, activation='linear')(x)
    #
    # model = Model(inputs=base_model.input, outputs=output_layer)

    model = get_efficient_unet_b0((256, 256, 3), pretrained=True)

    model.compile(optimizer=Adam(0.000001), loss='mse')

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    images_names = []

    index = -1
    for image_id, image_info in images_info.items():
        for annotation_info in annotations_info:
            if image_id == annotation_info.get("image_id"):
                index += 1
                annotation_info = annotations_info[index]
                img, lymphocyte_coords = extract_lymphocyte_coordinates(image_info, annotation_info)

                if img is None:
                    break

                img = cv2.resize(img, (256, 256))

                # cv2.imshow('Image', img)
                print(f'Lymphocyte Coordinates for Image {image_id}: {lymphocyte_coords}')
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                img = img / 255.0
                # if index % 4 == 0:
                #     X_test.append(img)
                #     Y_test.append(np.array(lymphocyte_coords[:2]).reshape((1, 2)))
                #     images_names.append(image_info['file_name'][9:])
                # else:
                X_train.append(img)
                Y_train.append(np.array(lymphocyte_coords[:2]).reshape((1, 2)))
                images_names.append(image_info['file_name'][9:])
                # break


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, Y_train, epochs=2000, batch_size=32, validation_data=(X_train, Y_train),
              callbacks=[early_stopping])
    y_pred = model.predict(X_train)
    print(y_pred)

    # output_file = 'predictions3.json'
    # predictions = [{'image_name': name, 'predicted_coordinates': list(coord)} for name, coord in zip(images_names, y_pred.tolist())]
    # with open(output_file, 'w') as f:
    #     json.dump(predictions, f)

