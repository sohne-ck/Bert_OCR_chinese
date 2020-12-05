#coding:utf-8

import csv
import os
from test_ocr import ocr_text
from bert_ner import bert_ner_text

i = 3
result = ocr_text(i)
# print(type(result))
# result = '有家烧烤店， 主营业:宵夜.烧烤.蛋炒饭'
# company_text = bert_ner_text(result)
