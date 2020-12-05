import json
import csv
idx = 0

nlabel = ['organization','company']

with open('Train_Data1.csv', 'a') as t:
    csv_write = csv.writer(t)
    # data_row = ["id","title","text","unknownEntities"]
    # csv_write.writerow(data_row)
    for line in open('dev.json','r'):
        label = json.loads(line)['label']
        text = json.loads(line)['text']
        label_list = []
        for i in label.items():
            if i[0] in nlabel:
                key = list(i[-1].keys())[0]
                label_list.append(key)

        if len(label_list)!=0:

            label_str = ''
            for l in label_list:
                label_str+=l
                label_str+=';'

            row_t = ['','',text,label_str]
            csv_write.writerow(row_t)
            print(text,"---label:",label_str)
            idx+=1

                # print(i.items)
    print("total number:",idx)
        # print(tweets)
