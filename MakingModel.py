# 내가 만든 MyVariable 파일을 가져옴.

from MyVariable import *

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, Sequential, load_model

# 모델 및 학습이력 저장을 위한 pickle 가져옴.
import pickle
import numpy as np

# train 데이터를 증식시키는 기능을 하는 함수
def bulge_data(train_dir, test_dir):
    # 이미지 어떻게 증식할 건지 설정. 모두의 딥러닝 307쪽 참고.
    # rescale : 0~255 값을 가지는 이미지 파일을 255로 나누어 0~1로 정규화 해줌. 혼자 튀는 값이 없도록!
    # rotation_range : 이미지 회전
    # width_shift_range : 이미지 수평으로 평행 이동
    # height_shift_range : 이미지 수직으로 평행 이동
    # shear_range : 이미지 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시킴. 일종의 비틀기 작업.
    # zoom_range : 이미지를 확대함.
    train_data_gen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=15, width_shift_range=0.1, # 기존엔 rotation 10
                                        height_shift_range=0.1, zoom_range=0.3,
                                        shear_range=0.2, horizontal_flip=True)
    print(f"train_data : {train_data_gen}")
    # test 파일은 데이터 증식이 필요 없음.
    test_data_gen = ImageDataGenerator(rescale=1./255)
    print(f"test_data : {test_data_gen}")

    # 실제 파일 가져와서 위의 데이터 증식적용.
    # flow_from_directory() : 폴더명을 category로 간주하고 label을 자동으로 생성. 폴더 하위에 있는 파일들을 다 label 붙여줌.
    train_data = train_data_gen.flow_from_directory(train_dir, batch_size=6,
                                                    color_mode='rgb', shuffle=True, class_mode='categorical',
                                                    target_size=(img_width,img_height))
    test_data = test_data_gen.flow_from_directory(test_dir, batch_size=10,
                                                    color_mode='rgb', shuffle=True, class_mode='categorical',
                                                    target_size=(img_width,img_height))

    # class (category)가 잘 분류되었는지 확인.
    print("train 데이터의 분류 :", train_data.class_indices.items())
    print("test 데이터의 분류 :", test_data.class_indices.items())

    return train_data, test_data


# VGG16 모델 가져오는 함수. 내 모델과 비교하려고 가져옴.
def make_origin_model():

    # 가중치 가져오기 or 초기화하기
    # vgg_model = VGG16(weights='imagenet')
    vgg_model = VGG16(weights=None, input_shape=(img_width, img_height, 3))
    return vgg_model

# 모델 직접 만들자. 가중치는 없음.
def make_transConv_model(dense):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    # imagenet 가중치 가져오기
    base_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("block5_pool").output)

    model = Sequential()
    model.add(base_model)

    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(GlobalAveragePooling2D())

    # 분류기 추가
    tf.keras.initializers.GlorotNormal(seed=None)
    initializer = tf.keras.initializers.GlorotNormal()

    model.add(Dense(1024, activation='relu', kernel_initializer=initializer))  # 분류기 부분 가중치를 초기화하기 위해서 kernel_initializer
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.5))
    model.add(Dense(dense, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.25))
    model.add(Dense(5, activation='softmax'))


    return model


# Final_Pro_T_GAP_D20_Loss(cross)_Adam1e-5_nostep_dropout025_xavier1_(200)

# top 수정하는 모델을 만드는 함수
def make_model(dense):
    # 기존의 VGG16 모델 가져오기.
    # weights='imagenet': imagenet으로 학습한 가중치를 그대로 가져옴.
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    # base_model.summary()

    # 전이학습으로 나만의 모델 만들기
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    # 분류기 추가
    tf.keras.initializers.GlorotNormal(seed=None)
    initializer = tf.keras.initializers.GlorotNormal()

    model.add(Dense(1024, activation='relu', kernel_initializer=initializer)) # 분류기 부분 가중치를 초기화하기 위해서 kernel_initializer
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.5))
    model.add(Dense(dense, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.25))
    model.add(Dense(5, activation='softmax'))

    model.summary()
    # 모델 컴파일
    # Adam(2e-5) : 학습률을 낮게 설정하여 pre-trained weights를 조금씩 업데이트해줌.
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-5),
                    metrics=['accuracy'])
    # loss='categorical_crossentropy'
    # loss='mean_squared_error'

    return model

# fine tuning
def make_model_FineTuning(dense):
    # make 모델로 완성한 하단 부분 가져오는 함수
    # top_model = load_model TODO 여기 LOAD하는 거 TOP 수정한 내 모델을 LOAD하는지 아닌지 체크.
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    base_model.trainable = True # base model인 vgg16의 가중치를 학습시키겠다(unfreeze). 그럼 기존 vgg16의 가중치가 변할 것.
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True # unfreeze
        if set_trainable:
            layer.trainable = True # unfreeze
        else:
            layer.trainable = False # freeze

    # base_model.summary()

    # 전이학습으로 나만의 모델 만들기
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    # 분류기 추가
    tf.keras.initializers.GlorotNormal(seed=None)
    initializer = tf.keras.initializers.GlorotNormal()

    model.add(Dense(1024, activation='relu', kernel_initializer=initializer)) # 분류기 부분 가중치를 초기화하기 위해서 kernel_initializer
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.5))
    model.add(Dense(dense, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.25))
    model.add(Dense(5, activation='softmax'))

    model.summary()
    # 모델 컴파일
    # Adam(2e-5) : 학습률을 낮게 설정하여 pre-trained weights를 조금씩 업데이트해줌.
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-5),
                    metrics=['accuracy'])
    # loss='categorical_crossentropy'
    # loss='mean_squared_error'

    return model

# train하고 train한 모델 및 학습이력을 저장하는 함수
def train(model, name, x, y):
    save_file_name = f"./model/{name}"
    hist_name = "hist_" + save_file_name.split("/")[2][:-3]

    # checkpoint = ModelCheckpoint(save_file_name, monitor='val_loss',
    #                                 verbose=1, save_best_only=True, model='auto')
    # earlystopping = EarlyStopping(monitor='val_loss', patience=5)

    # history = model.fit_generator(x, epochs=my_epoch, validation_data=y, callbacks=[checkpoint, earlystopping], verbose=2)
    history = model.fit_generator(x, epochs=my_epoch, validation_data=y)


    # 모델 저장
    model.save(save_file_name)

    # 학습이력 저장
    with open(f'./model/{hist_name}.pickle', 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == "__main__":
    # 1. --- 데이터 가져오기 및 전처리 하기 --- #
    # train 데이터와 validation 데이터가 있는 곳을 변수에 지정해줌.
    train_dir = './Data/train'
    test_dir = './Data/validation'

    # bulge_data 함수의 인자로 해당변수 넘겨주어서 데이터를 부풀리는 작업을 시행함.
    train_data, test_data = bulge_data(train_dir, test_dir)

    # 2. --- 모델 생성 및 학습하기 --- #
    # 모델 생성 및 학습을 위한 코드. 이미 만들어놓았기에 주석처리함.
    #  모델 생성
    new_model = make_transConv_model(dense)
    # print('before freeze, len of trainable_weight', len(model.trainable_weights))

    for layer in new_model.layers[:1]:
        layer.trainable = False

    new_model.summary()

    new_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-5),
                  metrics=['accuracy'])

    # 학습
    train(new_model, model_name, train_data, test_data)

    # 3. --- vgg16 기존 모델 구조 가져와서 학습 진행 ---
    origin_model = make_origin_model()
    origin_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-5),
                  metrics=['accuracy'])

    print(origin_model.summary())

    train(origin_model, model_name, train_data, test_data)


    model = load_model(f"./model/{model_name}")
    print(model.summary())

    train_score = model.evaluate_generator(train_data)
    print(f"train accuracy : {train_score}")

    test_score = model.evaluate_generator(test_data)
    print(f"test accuracy : {test_score}")


    # 3. --- 기존 모델 가져오기 --- #

