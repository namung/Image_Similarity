from MyVariable import *
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob

def ipt_hist_model(m_name, hist_name):
    """
    훈련된 모델과 그 학습이력을 가져옴.

    :param m_name: 모델 파일명
    :param hist_name: 학습이력 파일명
    :return: model과 histroy를 반환.
    """
    model = load_model(f"./model/{m_name}")
    hist = pickle.load(open(f"./model/{hist_name}", "rb"))
    return model, hist

def save_feature_matrix(name, matrix):
    """
    모델의 feature 저장하는 함수

    :param name: 저장할 feature 파일명
    :param model_name: 모델 이름.
    :param matrix: 모델이 뽑은 fueature matrix. 보통 model.predict()를 하면 feature를 뽑는다.
    :return: 없음! feature matrix를 pickle 파일로 저장함.
    """
    with open(f'./vgg16_feature/{name}.pickle', 'wb') as f:
        pickle.dump(matrix, f)

def show_performance_graph(hist, type, ylim_num):
    """
    모델의 loss, accuracy와 같은 성능을 보여주는 함수.

    :param model_name: 모델의 이름
    :param hist: 학습이력 이름.
    :param type: "loss" 그래프인지 "accuracy" 그래프인지 타입 적는 곳. string으로 해당 글자를 적어주면 됨.
    :return: 없음. 모델의 loss, accuracy 그래프를 보여줌.
    """
    plt.title(f"{type} trend")
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel(f'{type}')
    plt.xlim([0, my_epoch])
    plt.ylim([0, ylim_num])
    plt.plot(hist[f'{type}'], label='train')
    plt.plot(hist[f'val_{type}'], label='validation')
    plt.legend(loc='best')
    plt.show()

def prep_variableImg(image):
    """
    이미지 파일 가져와서 이미지를 전처리하는 함수.
    인자는 image = load_img(image, target_size=(img_width, img_height)) 한 것을 넣어야 함.
    위처럼 이미지를 load 한 것이 Image 객체.

    :param image: 전처리할 이미지의 <class 'PIL.Image.Image'> 타입의 객체를 넣어줌. 즉, 이미지 객체를 넣어줌.
    :return: 전처리 완료한 numpy.array를 반환.
    """
    load_image = load_img(image, target_size=(img_width, img_height))
    img_array = img_to_array(load_image)
    # dst_img = np.expand_dims(img_array, axis=0) # 배치 차원을 추가함. # dst_img = destination  ---> 이거 필요없을지도? 나중에 확인해보기.
    dst_img = img_array/255.

    return dst_img

def prep_oneImg(image):
    """
    이미지 파일 가져와서 이미지를 전처리하는 함수.
    인자는 image = load_img(image, target_size=(img_width, img_height)) 한 것을 넣어야 함.
    위처럼 이미지를 load 한 것이 Image 객체.

    :param image: 전처리할 이미지의 <class 'PIL.Image.Image'> 타입의 객체를 넣어줌. 즉, 이미지 객체를 넣어줌.
    :return: 전처리 완료한 numpy.array를 반환.
    """
    load_image = load_img(image, target_size=(img_width, img_height))
    img_array = img_to_array(load_image)
    dst_img = np.expand_dims(img_array, axis=0) # 배치 차원을 추가함. # dst_img = destination  ---> 이거 필요없을지도? 나중에 확인해보기.
    dst_img = dst_img/255.

    return dst_img

def show_variable_pred(pred, img_list):
    """
    여러 이미지를 놓고 모델보고 정답을 맞춰보도록해서 그 결과를 한 이미지로 반환하는 함수.
    :param model_name: 사용할 모델 이름.
    :param pred: 이미지를 모아놓은 리스트 배열을 보고, 특징만 뽑은 feature가 들어가는 곳. model.predict의 반환결과가 들어감.
    :param img_list: 화면에 표시할 이미지를 모아놓은 list. image를 load 하여 전처리 다 한 이미지를 리스트에 넣는데 그 리스트를 넣음.
    :return: 반환 없음. 예측 결과 이미지를 보여줌.
    """
    plt.figure(figsize=(15, 10))
    plt.suptitle("our model's predict", fontsize=20)

    for i in range(len(img_list)): # len(img_list): 예측 개수 반환. len(pred)와 똑같은 값 반환.
        num = 5 # 몇 행으로 표시할지.
        plt.subplot(num, int(len(img_list)/num), i + 1) # (행, 열, 몇번째에 위치시킬지)
        prediction = str(class_names[np.argmax(pred[i])])
        probility = f'{max(pred[i]) * 100 : .2f}' # 가능성
        title_str = prediction + ", " + probility + "%"
        plt.axis('off') # 서브플롯의 축을 끔.
        plt.title(title_str)
        plt.imshow(img_list[i]) # 여러 이미지 볼 때
    plt.show()

def show_one_pred(pred, image):
    """
    하나의 이미지를 보고, 그 이미지가 무엇인지 예측하는 함수
    :param pred: feature. model.predict() 했을 때 반환되는 값.
    :param image: Image 객체.
    :return: 없음. 모델이 사진을 보았을 때 무엇인지 보여주는 함수.
    """
    prediction = str(class_names[np.argmax(pred)])

    probility = f'{max(pred[0]) * 100 : .2f}'
    plt.figure(figsize=(8, 8))
    plt.title(f"[our model] this name: {prediction}, {probility} %")
    plt.imshow(image)
    plt.show()

def similarity_image(img, cos_df_t, rank_num, db_img_path):
    """
    입력한 하나의 이미지 파일과 특정 폴더 안에 있는 이미지 파일 전부를 비교해서 유사도를 비교하는 함수.

        :param img: 하나의 이미지 파일.
        :param cos_df_t: cosine 벡터.
        :return: 없음. 선택한 이미지 하나를 보여주고 유사한 이미지
    """
    # print('-' * 30)
    # print(f'selected product: {img}')
    # image = load_img(img, target_size=(img_width, img_height))
    # plt.imshow(image)
    # plt.show()

    print('-' * 30)
    print('similar products:')

    # cos_df_t : df 타입. ./Data/test/chair2.jpg 열을 가지고 test 파일의 모든 파일과 유사도 비교한 값이 있는 df.
    # cos_df_t[one_img].sort_values(ascending=False) : 코사인 유사도 값에 대해 내림차순으로 정렬.
    # cosine_img : 선택한 폴더 안의 전체 파일에 대해서 선택한 이미지 파일을 제외한 나머지에 대해 most_similar 값을 포함해서 슬라이싱 함. 슬라이싱 해서 그 index를 가져오는데, index는 확장자를 제외한 파일명임.
    # most_similar = 최대 몇개의 유사한 이미지를 보여줄 건지 그 개수를 넣는 변수. MyVariable.py에 있음.
    # cosine_img = cos_df_t[one_img].sort_values(ascending=False)[1:rank_num + 1].index
    cosine_img = cos_df_t[one_img].sort_values(ascending=False)[1:rank_num + 1].index
    # cosine_img_score = cos_df_t[one_img].sort_values(ascending=False)[1:rank_num + 1]
    cosine_img_score = cos_df_t[one_img].sort_values(ascending=False)[1:rank_num + 1]
    print(cosine_img_score)

    # # most_similar 이미지 보여주기.
    # for i in range(len(cosine_img)):
    #     image_cosine = load_img(db_img_path[:-1] + cosine_img[i] + ".jpg", target_size=(img_width, img_height))
    #     plt.imshow(image_cosine)
    #     plt.show()

    # 유사도 평균 구하기.
    total = 0
    for i in cosine_img_score:
        total += i
    print(f"rank {rank_num}개 유사도 평균:", total/len(cosine_img_score))

def make_origin_model():
    vgg_model = VGG16(weights='imagenet')
    # imagenet 가중치 가져오기
    model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("flatten").output)
    return model

if __name__ == "__main__":
    # 0. MyVariable.py 먼저 확인.


    # 1. --- 학습한 모델과 학습 이력 가져오기 ---

    # ipt_hist_model() 함수는 기존에 만들어 둔 모델과 학습 이력을 import 해오는 함수.
    # 학습 모델은 our_model 변수에, 학습 이력은 hist 변수에 저장해둠.
    # model_name, hist_name 는 MyVariable.py 에 있음.
    our_model, hist = ipt_hist_model(model_name, hist_name)
    our_model.summary()

    # 2. --- 모델의 특징추출기만 가져오기. ---

    feature_model = Model(inputs=our_model.input, outputs=our_model.get_layer("global_average_pooling2d").output)
    feature_model.summary()

    #####      vgg16 모델 특징부분 가저와서 불러와서 featrue 저장
    # vgg16_feature = make_origin_model()


    # 3. --- 학습한 모델의 학습 이력 확인하기. ---

    # loss 확인.
    # 학습 이력을 가져와서 학습하는 동안에 loss 값이 얼마나 떨어졌는지 그래프로 확인.
    show_performance_graph(hist, "loss", 4)

    # acurracy 확인.
    # 학습 이력을 가져와서 학습하는 동안에 accuracy가 얼마나 올라갔는지 그래프로 확인.
    show_performance_graph(hist, "accuracy", 1.0)


    # 4. --- 생성한 모델의 성능 확인 --- #

    # 1) 더 많은 데이터로 모델 확인하기
    # 테스트 폴더 가져옴.
    tmp_file_list2 = glob.glob("./Data/test/*")
    # tmp_file_list2 = glob.glob("./Data/train/*")

    class_list = []
    pred_list = []

    for i, v in enumerate(tmp_file_list2): # x 5
        print(f" {i+1} / {len(tmp_file_list2)} 폴더 확인 중...")
        idx = i

        test_dst_list = []
        file_list = glob.glob(v + "/*")
        print(file_list)

        # for i in range(len(file_list)):  # 한 폴더 안의 총 50 파일.
        #     dst_img = prep_img(file_list[i])
        #     test_dst_list.append(dst_img)
        #
        # for i in range(len(test_dst_list)): # 총 50개.
        #     prepArray = np.array(test_dst_list[i])
        #     prepArray = np.expand_dims(prepArray, axis=0)
        #     our_pred = our_model.predict(prepArray)
        #     pred_list.append(str(class_names[np.argmax(our_pred)]))

        for i in range(len(file_list)):  # 한 폴더 안의 총 50 파일.
            dst_img = prep_oneImg(file_list[i])
            our_pred = our_model.predict(dst_img)
            pred_list.append(str(class_names[np.argmax(our_pred)]))
            class_list.append(class_names[idx])

    print(class_list)
    print(classification_report(class_list, pred_list))


    # 모델이 처음보는 자료(test 데이터)를 가져와서 예측하는 거 이미지로 보여주는 걸로 모델의 성능 확인하기.
    # 여러 테스트 이미지를 한 번에 가져와서 모델이 예측하는 것과 하나의 사진을 보고 예측하는 것, 2개를 시행해 볼 것.

    # 1) 여러 테스트 이미지를 한 번에 가져와서 모델이 예측해보기.
    # (1) test 파일 가져오기.
    # test_img_list 변수에 test 폴더 안에 있는 모든 파일의 이름을 리스트 형태로 가져옴. glob 함수를 이용하면 리스트 형태로 가져와짐.
    # test_img_list 변수 안에 있는 이미지 파일을 하나씩 가져와서 전처리를 시행한 후, test_dst_list에 하나씩 넣어줌.
    file_list = glob.glob("./Data/test/table/*")
    test_dst_list = [] # dst: destination. 목적지.

    # (2) 반복문으로 이미지 파일 전처리하고 새로운 리스트에 전처리 완료한 파일 넣기.
    for i in range(len(file_list)):
        dst_img = prep_img(file_list[i])
        test_dst_list.append(dst_img)
    # print(file_list)

    # (3) 학습한 모델로 test 이미지 파일이 무엇인지 예측해보기.
    # 이미지 리스트를 보며 특징값을 하나씩 가져와서 np.array에 한 이미지씩 그 특징값을 저장한 다음, our_pred 변수에 저장.
    # model.predict() --> 이 함수를 이용해서 내 모델을 이용해 해당 이미지 파일의 특징값을 들고 옴!!!!
    our_pred1 = our_model.predict(np.array(test_dst_list))


    # (4) 여러 이미지 예측 시행해서 그림으로 보여주기.
    show_variable_pred(our_pred1, test_dst_list)


    # # 3) 한 개의 이미지 전처리 및 예측하기 & 데이터 특징 뽑기.
    # # (1)테스트할 파일을 하나만 가져와서 one_test_file 변수에 할당하기.
    # # one_img : MyVariable.py 파일에서 확인!. 이미지 파일 하나를 선택해서 가져옴.
    # img1 = "./Data/temp1_test/bed1.jpg"
    # load_image = load_img(img1, target_size=(img_width, img_height))
    # dst_img = img_to_array(load_image)
    # dst_img = np.expand_dims(dst_img, axis=0) # 하나의 이미지에 대해서는 배치 차원이 추가되어야 함.
    # dst_img /= 255.
    # our_pred2 = our_model.predict(dst_img)
    #
    # # (2) 예측하기
    # show_one_pred(our_pred2, load_image)


    # 5. --- 모델이 이미지를 보고 뽑은 feature 값을 따로 저장하기. ---

    # (1) 각각 이미지 파일에 대한 feature 저장하기. 나중에 db 저장해서 사용하기 위함.
    # 이 파일을 불러와서 코사인 유사도 비교하는 방법도 있긴한데 우리는 일단은 그냥 한 matrix로 저장해서 사용할 것.

    # for 문 이용하여 해당 폴더에 있는 테스트할 이미지의 feaure를 한번에 저장하기.
    # f_name : 확장자명을 제외한 파일명. MyVariable.py 참고.
    # test_dst_list : 199라인 참고. 전처리 완료한 이미지의 리스트를 가지고 있음.
    img_path = "./Data/temp1_test _2/*"
    file_list = glob.glob(img_path)
    for i in range(len(file_list)):
        file = f"{file_list[i]}" # 파일을 하나만 가져옴.
        f_name = [file.split(".")[1].split(".")[0].split("\\")[1] for file in file_list]

        name = f_name[i] # 해당 파일의 이름을 가져옴.
        dst_img = prep_oneImg(file)
        our_pred3 = feature_model.predict(np.array(dst_img))
        save_feature_matrix(name, our_pred3)

    # # (2) feature load 하기
    # path = './features/*'
    # fileList = glob.glob(path)
    # feature_list = []
    # for v in range(len(fileList)):
    #     print(fileList[v])
    #     feature = pickle.load(open(fileList[v], "rb"))
    #     feature_list.append(feature)


    # 6. --- 모델이 테스트 이미지를 보고 각 이미지간의 유사도 비교 ---
    # # (1) 이미지 하나에 대한 batch_size와 feature 개수 확인해보기.
    # image_1 = tmp_file_list[5]
    # processed_image1 = prep_img(image_1)
    # dst_img1 = np.expand_dims(processed_image1, axis=0)
    # print('image_batch_size=', dst_img1.shape)
    #
    # img_feature = feature_model.predict(dst_img1)
    # print('number of image features: ', img_feature.size)
    # # print(img_feature)

    # (2) 이미지 여러개를 전처리해서 한 리스트에 담기.
    # 빈리스트 준비.
    image_list = []
    # file_list는 MyVariable.py 파일에 있음. test 폴더 속 파일 전체를 하나씩 가져와서 전처리 후에 리스트에 넣기.
    for f in db_file_list: # 5개
        db_file_name = f
        dst_img = prep_oneImg(db_file_name)
        image_list.append(dst_img)

    images = np.vstack(image_list) # 리스트 안에 리스트끼리 세로로 합침. 하나의 matrix 완성.     # hstack -> 가로로 합치기

    # (3) 전처리된 이미지들을 합쳐놓은 matrix. 이걸로 한 번에 feature 뽑기!
    # img_feature = feature_model.predict(images)
    img_feature = feature_model.predict(images)
    print("feature 뽑을 이미지 개수 :", len(img_feature))

    # (4) feaure 이용해서 유사도 비교하기.
    # 유사도 비교할 기준이 되는 이미지 선택하기. 그 기준이 되는 이미지가 one_img.
    # one_img 변수는 MyVariable.py 에서 확인.
    processed_image2 = prep_oneImg(one_img) # 전처리 완료
    # 기준이 되는 이미지 feature 뽑기.
    one_feature = feature_model.predict(processed_image2)

    # 기준이 되는 이미지와 다른 전체 이미지의 유사도 비교하기한 값을 cosine 변수에 저장.
    # cosine은 matrix 형태로, 기준이 되는 이미지와 나머지 전체 테스트 폴더의 이미지에 대한 코사인 유사도 값이 저장되어 있음.
    cosine = cosine_similarity(one_feature, img_feature) # cosine : <class 'numpy.ndarray'>
    # print(cosine)

    # numpy.ndarray 타입을 데이터프레임으로 바꿔줌. 정제해서 알아보기 쉽게 사용할 것.
    # f_name : 확장자명을 제외한 파일명. MyVariable.py 참고.
    cos_df = pd.DataFrame(cosine, columns=f_name) # columns 명을 알아보기 쉽게 파일이름으로 바꿈.
    cos_df_t = cos_df.transpose() # 행열 전환
    cos_df_t.columns = [one_img]

    # 기준이 되는 이미지와 나머지 유사한 이미지는 무엇인지, 유사도는 어느정도 되는지, rank 매겨서 순서대로 보여주는 함수.
    # most_similar
    similarity_image(one_img, cos_df_t, rank_num, db_img_path)