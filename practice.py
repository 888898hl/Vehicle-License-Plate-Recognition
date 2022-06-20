#print(contours[0][2][0][0])
import cv2
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imutils
car_plate_w, car_plate_h = 180, 36  # 车牌宽和高 180
char_w,char_h = 20,20
cur_dir = sys.path[0]
char_model_path = os.path.join(cur_dir,'./carIdentityData/model/char_recongnize/model.ckpt-590.meta')#字符识别模型
char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']
# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()
def plt_show0(img):#plt显示彩色图
    # cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()
def show_pic(img):
    plt.imshow(img,"gray")
    plt.show()
frame = cv2.imread("car4.jpg")
color_oringe=frame.copy()#彩色原图
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#-------------------提取蓝色区域
lower_blue = np.array([100, 47, 47])
upper_blue = np.array([124, 255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue) #蓝色掩模,除了蓝色区域其他都是黑色,此时mask通道只有一个通道
res = cv2.bitwise_and(frame, frame, mask = mask)
#用mask图片即可
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
# close_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 形态学滤波，找封闭的框图

oringe_img=mask.copy()  #二值化图原图
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))#核越大，腐蚀＼膨胀越粗糙
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelX,iterations = 5)#先膨胀再腐蚀，用于去除前景中的黑点
show_pic(mask)
num=0;
_,contours, heriachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i,contour in enumerate(contours):  # contours是轮廓点集
    print(i)
    x,y,w,h = cv2.boundingRect(contour)#获取图片下的左上角坐标
    if(2.5<w/h and w/h<5 and w*h>40000 ):    #正常车牌在160000数量级
        cv2.drawContours(frame, contours, i, (0, 255, 255), 3)
        num+=1#打印最后找到的轮廓数量
        print("area=",cv2.contourArea(contours[i]))
        plate = color_oringe[y:y + h, x:x + w]
        m_car_plate = cv2.resize(plate, (car_plate_w, car_plate_h))  # 调整尺寸为后面CNN车牌识别做准备
        plt_show0(m_car_plate)  # 彩色图
    else:
        continue
print("num=",num)
show_pic(frame)  # 显示画了边框的图
#cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#cv2.line(img,(contours[0][0][0][0],contours[0][0][0][1]),(contours[0][1][0][0],contours[0][1][0][1]),(255,255,0),30)#(contours[0][0][0][0],contours[0][0][0][1]),(contours[0][1][0][0],contours[0][1][0][1])
# cv2.drawContours(copy1, contours, -1, (0, 255, 255), 2)  # 画黄色轮廓线，非直线，而是轮廓线
#cv2.imshow("copy",img)


# 左右切割
def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left,area_right,char_left,char_right= 0,0,0,0
    img_w = plate.shape[1]

    # 获取车牌每列边缘像素点个数
    def getColSum(img,col):
        sum = 0
        for i in range(img.shape[0]):
            sum += round(img[i,col]/255)
        return sum;

    sum = 0
    for col in range(img_w):
        sum += getColSum(plate,col)
    # 每列边缘像素点必须超过均值的60%才能判断属于字符区域
    col_limit = 0#round(0.5*sum/img_w)
    # 每个字符宽度也进行限制
    charWid_limit = [round(img_w/12),round(img_w/5)]
    is_char_flag = False

    for i in range(img_w):
        colValue = getColSum(plate,i)
        if colValue > col_limit:
            if is_char_flag == False:
                area_right = round((i+char_right)/2)
                area_width = area_right-area_left
                char_width = char_right-char_left
                if (area_width>charWid_limit[0]) and (area_width<charWid_limit[1]):
                    char_addr_list.append((area_left,area_right,char_width))
                char_left = i
                area_left = round((char_left+char_right) / 2)
                is_char_flag = True
        else:
            if is_char_flag == True:
                char_right = i-1
                is_char_flag = False
    # 手动结束最后未完成的字符分割
    if area_right < char_left:
        area_right,char_right = img_w,img_w
        area_width = area_right - area_left
        char_width = char_right - char_left
        if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))
    return char_addr_list

def get_chars(car_plate):
    img_h,img_w = car_plate.shape[:2]
    h_proj_list = [] # 水平投影长度列表
    h_temp_len,v_temp_len = 0,0
    h_startIndex,h_end_index = 0,0 # 水平投影记索引
    h_proj_limit = [0.2,0.8] # 车牌在水平方向得轮廓长度少于20%或多余80%过滤掉
    char_imgs = []

    # 将二值化的车牌水平投影到Y轴，计算投影后的连续长度，连续投影长度可能不止一段
    h_count = [0 for i in range(img_h)]
    for row in range(img_h):
        temp_cnt = 0
        for col in range(img_w):
            if car_plate[row,col] == 255:
                temp_cnt += 1
        h_count[row] = temp_cnt
        if temp_cnt/img_w<h_proj_limit[0] or temp_cnt/img_w>h_proj_limit[1]:
            if h_temp_len != 0:
                h_end_index = row-1
                h_proj_list.append((h_startIndex,h_end_index))
                h_temp_len = 0
            continue
        if temp_cnt > 0:
            if h_temp_len == 0:
                h_startIndex = row
                h_temp_len = 1
            else:
                h_temp_len += 1
        else:
            if h_temp_len > 0:
                h_end_index = row-1
                h_proj_list.append((h_startIndex,h_end_index))
                h_temp_len = 0

    # 手动结束最后得水平投影长度累加
    if h_temp_len != 0:
        h_end_index = img_h-1
        h_proj_list.append((h_startIndex, h_end_index))
    # 选出最长的投影，该投影长度占整个截取车牌高度的比值必须大于0.5
    h_maxIndex,h_maxHeight = 0,0
    for i,(start,end) in enumerate(h_proj_list):
        if h_maxHeight < (end-start):
            h_maxHeight = (end-start)
            h_maxIndex = i
    if h_maxHeight/img_h < 0.5:
        return char_imgs
    chars_top,chars_bottom = h_proj_list[h_maxIndex][0],h_proj_list[h_maxIndex][1]

    plates = car_plate[chars_top:chars_bottom+1,:]
    cv2.imwrite('./carIdentityData/opencv_output/car.jpg',car_plate)
    cv2.imwrite('./carIdentityData/opencv_output/plate.jpg', plates)
    char_addr_list = horizontal_cut_chars(plates)

    for i,addr in enumerate(char_addr_list):
        char_img = car_plate[chars_top:chars_bottom+1,addr[0]:addr[1]]
        char_img = cv2.resize(char_img,(char_w,char_h))
        char_imgs.append(char_img)
    return char_imgs

def extract_char(car_plate):
    gray_plate = cv2.cvtColor(car_plate,cv2.COLOR_BGR2GRAY)
    ret,binary_plate = cv2.threshold(gray_plate,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    char_img_list = get_chars(binary_plate)
    return char_img_list
def cnn_recongnize_char(img_list,model_path):
    g2 = tf.Graph()
    sess2 = tf.Session(graph=g2)
    text_list = []

    if len(img_list) == 0:
        return text_list
    with sess2.as_default():
        with sess2.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess2, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            net2_x_place = graph.get_tensor_by_name('x_place:0')
            net2_keep_place = graph.get_tensor_by_name('keep_place:0')
            net2_out = graph.get_tensor_by_name('out_put:0')

            data = np.array(img_list)
            # 数字、字母、汉字，从67维向量找到概率最大的作为预测结果
            net_out = tf.nn.softmax(net2_out)
            preds = tf.argmax(net_out,1)
            my_preds= sess2.run(preds, feed_dict={net2_x_place: data, net2_keep_place: 1.0})

            for i in my_preds:
                text_list.append(char_table[i])
            return text_list
 #字符提取
char_img_list = extract_char(m_car_plate)#其中car_plate_list[0]就是车牌区域的彩色图片

#CNN字符识别
text = cnn_recongnize_char(char_img_list,char_model_path)
print(text)


# # 图像去噪灰度处理
# def gray_guss(image):
#     image = cv2.GaussianBlur(image, (3, 3), 0)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     return gray_image
# #---------------------------------------------------------下面开始进行字符识别-----------------------
# #车牌字符分割
# # 图像去噪灰度处理
# gray_image = gray_guss(plate)
# #图像阈值化操作——获得二值化图
# ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
# plt_show(image)
# #膨胀操作，使“苏”字膨胀为一个近似的整体，为分割做准备
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# image = cv2.dilate(image, kernel)
# #plt_show(image)
# cv2.imshow("image",image)
# #------------------------------------
# # 查找轮廓
# contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# words = []
# word_images = []
# #对所有轮廓逐一操作
# for item in contours:
#     word = []
#     rect = cv2.boundingRect(item)
#     x = rect[0]
#     y = rect[1]
#     weight = rect[2]
#     height = rect[3]
#     word.append(x)
#     word.append(y)
#     word.append(weight)
#     word.append(height)
#     words.append(word)
# # 排序，车牌号有顺序。words是一个嵌套列表
# words = sorted(words,key=lambda s:s[0],reverse=False)
# i = 0
# #word中存放轮廓的起始点和宽高
# for word in words:
#     # 筛选字符的轮廓
#     if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 3.5)) and (word[2] > 25):
#         i = i+1
#         splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
#         word_images.append(splite_image)
#         print(i)
# print(words)
#
# for i,j in enumerate(word_images):
#     plt.subplot(1,7,i+1)
#     plt.imshow(word_images[i],cmap='gray')
# plt.show()
# #-------------------------------------------
# #模版匹配
# # 准备模板(template[0-9]为数字模板；)
# template = ['0','1','2','3','4','5','6','7','8','9',
#             'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
#             '藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁','蒙','闽','宁',
#             '青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙']
#
# # 读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表
# def read_directory(directory_name):
#     referImg_list = []
#     for filename in os.listdir(directory_name):
#         referImg_list.append(directory_name + "/" + filename)
#     return referImg_list
#
# # 获得中文模板列表（只匹配车牌的第一个字符）
# def get_chinese_words_list():
#     chinese_words_list = []
#     for i in range(34,64):
#         #将模板存放在字典中
#         c_word = read_directory('refer1/'+ template[i])
#         chinese_words_list.append(c_word)
#     return chinese_words_list
# chinese_words_list = get_chinese_words_list()
#
# # 获得英文模板列表（只匹配车牌的第二个字符）
# def get_eng_words_list():
#     eng_words_list = []
#     for i in range(10,34):
#         e_word = read_directory('./refer1/'+ template[i])
#         eng_words_list.append(e_word)
#     return eng_words_list
# eng_words_list = get_eng_words_list()
# # 获得英文和数字模板列表（匹配车牌后面的字符）
# def get_eng_num_words_list():
#     eng_num_words_list = []
#     for i in range(0,34):
#         word = read_directory('./refer1/'+ template[i])
#         eng_num_words_list.append(word)
#     return eng_num_words_list
# eng_num_words_list = get_eng_num_words_list()
# # 读取一个模板地址与图片进行匹配，返回得分
# def template_score(template,image):
#     #将模板进行格式转换
#     template_img=cv2.imdecode(np.fromfile(template,dtype=np.uint8),1)
#     template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
#     #模板图像阈值化处理——获得黑白图
#     ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
# #     height, width = template_img.shape
# #     image_ = image.copy()
# #     image_ = cv2.resize(image_, (width, height))
#     image_ = image.copy()
#     #获得待检测图片的尺寸
#     height, width = image_.shape
#     # 将模板resize至与图像一样大小
#     template_img = cv2.resize(template_img, (width, height))
#     # 模板匹配，返回匹配得分
#     result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
#     return result[0][0]
# # 对分割得到的字符逐一匹配
# def template_matching(word_images):
#     results = []
#     for index,word_image in enumerate(word_images):
#         if index==0:
#             best_score = []
#             for chinese_words in chinese_words_list:
#                 score = []
#                 for chinese_word in chinese_words:
#                     result = template_score(chinese_word,word_image)
#                     score.append(result)
#                 best_score.append(max(score))
#             i = best_score.index(max(best_score))
#             # print(template[34+i])
#             r = template[34+i]
#             results.append(r)
#             continue
#         if index==1:
#             best_score = []
#             for eng_word_list in eng_words_list:
#                 score = []
#                 for eng_word in eng_word_list:
#                     result = template_score(eng_word,word_image)
#                     score.append(result)
#                 best_score.append(max(score))
#             i = best_score.index(max(best_score))
#             # print(template[10+i])
#             r = template[10+i]
#             results.append(r)
#             continue
#         else:
#             best_score = []
#             for eng_num_word_list in eng_num_words_list:
#                 score = []
#                 for eng_num_word in eng_num_word_list:
#                     result = template_score(eng_num_word,word_image)
#                     score.append(result)
#                 best_score.append(max(score))
#             i = best_score.index(max(best_score))
#             # print(template[i])
#             r = template[i]
#             results.append(r)
#             continue
#     return results
# word_images_ = word_images.copy()
# # 调用函数获得结果
# result = template_matching(word_images_)
# print(result)
# # "".join(result)函数将列表转换为拼接好的字符串，方便结果显示
# print( "".join(result))
#
cv2.waitKey(0)
