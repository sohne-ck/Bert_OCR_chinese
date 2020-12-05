import os

# import numpy as np
# from PIL import Image
from django.http import JsonResponse
import sys
sys.path.append('/home/mist/')
import connection

def upload(request):
    if request.method == 'POST':
        file = request.FILES.get('file', None)
        # file_name = request.FILES.get('file', None).name
        print(file)
        # print(file_name)
        if not file:
            return JsonResponse({'code': '300', 'message': '没有图片', 'data': ''}, safe=False)
        # file_path = os.path.abspath(os.path.dirname(os.getcwd())) + '\\ocr\\images\\'
        file_path = '/home/mist/ocr/image/images'
        # file_path = 'E:\\zrb\\software_competition\\ocr\\image\\images\\'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        destination = open(os.path.join(file_path, file.name), 'wb+')
        for chunk in file.chunks():
            destination.write(chunk)
        destination.close()
        # im = Image.open(file_path + file.name)
        # image = np.array(im)
        # print('image->', image.shape)

        # 调用算法
        filename = os.path.basename(os.path.splitext(file.name)[0])
        ret_str = connection.ocr_test(filename)
        return JsonResponse({'code': '200', 'message': '成功', 'data': str(ret_str)}, safe=False)
    else:
        return JsonResponse({'code': '300', 'message': '请使用POST请求', 'data': ''}, safe=False)

