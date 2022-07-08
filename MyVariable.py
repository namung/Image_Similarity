import glob

# 0. --- 파일을 실행할 때 사용할 여러 변수 지정. --- #
# resize 할 때 사용할 이미지 파일 크기 지정.
# VGG16은 입력 이미지 파일을 224*224로 받음. 차원은 4차원으로 보통 1*224*224*3으로 받는다. 1은 배치차원? 배치. 3은 RGB를 의미함.
img_width, img_height = 224, 224

# 학습시 epoch 지정.
my_epoch = 200

# 학습시 분류부분의 layer의 노드개수 지정.
dense = 20

# 모델 파일 이름에 쓸 변수인 mode. 그냥 단순히 이름 지정하는 용도.
mode = "GAP"

# 모델 파일 이름 생성.
# model_name = f"vgg16_{mode}_D{dense}_({my_epoch}).h5"
# model_name = f"Final_Pro_T_{mode}_D{dense}_Loss(cross)_Adam1e-5_nostep_dropout025_xavier1_({my_epoch}).h5"
model_name = f"Final_Pro_T_finetuning_and_no_Fillmode_and_differ_{mode}_D{dense}_Loss(cross)_Adam1e-5_nostep_dropout025_xavier1_({my_epoch}).h5"
# model_name = f"trans_add_conv_512(1)_D{dense}__dropout05_({my_epoch}).h5"
# model_name = f"trans_add_conv_64(1)_D{dense}__dropout05_({my_epoch}).h5"

# 모델 학습시 생성된 학습이력 파일에 해당 모델의 이름을 붙어줌.
hist = model_name.split(".")[0]
hist_name = f"hist_{hist}.pickle"

# 내 데이터의 category, 클래스를 지정함.
class_names = ["bed", "chair", "closet", "shelves", "table"]

# 내 db 위치 지정하기.
img_path = './Data/temp1_test/'

# 이미지 특징 가져올 때 필요한 함수.
# glob 함수로 테스트 이미지를 가져옴.
# 특정 폴더에서.
# tmp_file_list = glob.glob("./Data/test/*")
# tmp_file_list = glob.glob("./Data/temp1_test/table/*")

# 유사도 측정 시 train 폴더 파일
# db_img_path = "./Data/train/*"
# db_img_path = "./Data/temp1_test/chair/*"
# db_img_path = "./Data/temp1_test/bed/*"
# db_img_path = "./Data/temp1_test/closet/*"
# db_img_path = "./Data/temp1_test/shelves/*"
db_img_path = "./Data/temp1_test/table/*"
db_file_list = glob.glob(db_img_path) # 5개

# files = [img_path + x for x in os.listdir(img_path) if 'jpg' in x]
f_name = [file.split(".")[1].split(".")[0].split("\\")[1] for file in db_file_list]
# 유사한 아이템 몇개를 뽑을것인지
rank_num = len(f_name)
# 이미지 하나만 뽑아오는 거
# one_img = "./Data/temp1_test/chair/chair1.jpg"
# one_img = "./Data/temp1_test/bed/bed1.jpg"
# one_img = "./Data/temp1_test/closet/closet1.jpg"
# one_img = "./Data/temp1_test/shelves/shelves1.jpg"
one_img = "./Data/temp1_test/table/table1.jpg"