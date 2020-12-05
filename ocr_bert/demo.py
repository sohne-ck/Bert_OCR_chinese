# coding: utf-8

# In[3]:



from test_ocr import init_ocr_model,ocr_img
import csv
import os
import time



# 假设我们要写入的是以下两行数据
dir_path = os.getcwd()

# 文件头，一般就是数据名
fileHeader = ["图片", "商铺名称"]
# 写入数据

csvFile = open("company.csv", "a")
writer = csv.writer(csvFile)

# 写入的内容都是以列表的形式传入函数
writer.writerow(fileHeader)

csvFile.close()

ocr_detection_model, ocr_recognition_model, ocr_label_dict = init_ocr_model()

def ocr_test(i):
    UPLOAD_FOLDER = './uploads'
    IMAGE_FOLDER = './image'
    VIDEO_FOLDER = r'./video'
    FOND_PATH = '/ocr/STXINWEI.TTF'
    # for i in range(14,51):

    ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4','avi'])
    VIDEO_EXTENSIONS = set(['mp4', 'avi'])
    img_path = dir_path+"/ocr/image/images/" + str(i) + ".jpg"
    save_path = dir_path+"/ocr/image/test_img/"+ str(i) + ".jpg"
    result = ocr_img(img_path,save_path, ocr_detection_model, ocr_recognition_model, ocr_label_dict)
    return result

from bert_ner import bert_ner_text

def companyText(re,n):
    result_str = ''
    for i in range(0,len(re)):
        result_a = ''
        for j in range(0,len(re[i])):
            result_a+=re[i][j]
        result_str+=result_a
        result_str+=','

    company_text = bert_ner_text(result_str)
    print(company_text)
    d1 = [str(n)+'.jpg', company_text]
    csv_file = open("company.csv", "a")
    writer = csv.writer(csv_file)
    writer.writerow(d1)
    csv_file.close()
    





# In[9]:


for n in range(1,51):
    start_time = time.time()
    i = str(n)
    re = ocr_test(i)
    companyText(re,i)
    end_time = time.time()
    print("使用时间：",end_time-start_time,'s')


# In[10]:


import pandas as pd
csv2Excel = pd.read_csv('company.csv', encoding='utf-8')
csv2Excel.to_excel(r'company.xlsx', sheet_name='data')


# In[ ]:



# 导入模型权重

result_str = ''
for i in range(0,len(re)):
    result_a = ''
    for j in range(0,len(re[i])):
        result_a+=re[i][j]
    result_str+=result_a
    result_str+=','

print(result_str)

company_text = bert_ner_text(result_str)


# In[8]:


print(company_text)

