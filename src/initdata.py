from PIL import Image, ImageDraw, ImageFont
import codecs as cs
import pyprind
import pandas as pd
import os
import numpy as np
import argparse



def importData(path_data, num=50000, file_name='data'):
    #总共有50000个文件
    pbar = pyprind.ProgBar(num)
    #将字符串类标数值化
    labels = {"pos":1,"neg":0}
    data = pd.DataFrame()

#    for s in ("test","train"):
    for s in ("train",):
        for l in ("pos","neg"):
            left = num/2
            #获取电影评论的存放路径
            path = path_data + "/aclImdb/%s/%s"%(s,l)
            #遍历电影的评论文件
            for file in os.listdir(path):
                with open(os.path.join(path,file),"r",encoding="utf-8") as f:
                    #获取电影评论
                    txt = f.read()
                if left <= 0:
                    break
                #将电影评论和评论的情感信息存入到DataFrame中
                txt = txt.replace('<br /><br />', '\n')
                if(len(txt) > 2000):
                    continue

                left = left - 1
                data = data.append([[txt,labels[l]]],ignore_index=True)
                #更新进度条
                pbar.update()
    #设置列名
    data.columns = ["review","sentiment"]
    #打乱电影评论的顺序
    np.random.seed(0)
    data = data.reindex(np.random.permutation(data.index))
    #保存文件
    print(path_data + file_name + ".csv")
    data.to_csv(path_data + file_name + ".csv",index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Init dataset.')   
    parser.add_argument('--test', dest='test', action='store_true',
                        help='if test?')    

    args = parser.parse_args()
    if args.test:
        print("Using test mode...")
        importData('./datasets/', 10, 'minidata')
    else:
        importData('./datasets/', 5000)
