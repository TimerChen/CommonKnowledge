from PIL import Image, ImageDraw, ImageFont
import codecs as cs
import pyprind
import pandas as pd
import os
import numpy as np
import argparse
import re

def genImage(char, width=None, height=None, font=None):
    if font is None:
        size = 14
    else:
        size = font.size
    
    if width is None:
        width = len(char)
    if height is None:
        height = 1
    w_size, h_size = font.getsize('a')
    h_size+=1
    image = Image.new('1', (width*w_size, height*h_size), 'white')
    #width*w_size = 840
    #image = Image.new('1', (100, 100), 'white')
    draw = ImageDraw.Draw(image)
    if font is None:
        draw.text((0, 0), char, fill='black')
    else:
        draw.multiline_text((0, 0), char, font=font, spacing=1, fill='black')
    return image


scCheck = re.compile(r"[\u4e00-\u9fff]+")

def deal_text(text, font, width_limit = None):
    width, height = 0, 0
    #Remove multiple \n
    
    
    #Remove \n in one_line mode
    res = ""
    width = height = 1
    if width_limit is None:
        res = text.replace('\n', '')
        width = len(text)
    else:
        text = text.split('\n')
        width = width_limit
        for i in text:
            while len(i) > width_limit:
                j = width_limit
                while not i[j] == ' ':
                    j-=1
                res += i[:j] + '\n'
                i = i[j+1:]
                height += 1
                
            if len(i) > 0:
                res += i + '\n'
                height += 1
    #width,height=font.getsize(res)
    
    return res, width, height

def genPics(file_name = 'data.csv', path_data = '../datasets/', folder='pics'):
    data = pd.read_csv(path_data + file_name)
    width, height = 0, 0
    num = data.shape[0]
    print(num)
    reviews = list(data['review'])
    sentiment = list(data['sentiment'])
#    font = ImageFont.truetype('/usr/share/fonts/YaHeiConsolas.ttf',20)
    font_dir = '/usr/share/fonts/truetype/freefont/FreeMono.ttf'
    font_dir = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"
    font_dir = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    font = ImageFont.truetype(font_dir,20)
    
    pbar = pyprind.ProgBar(num)
    for i in range(num):
        #print(len(reviews[i]), sentiment[i])
        reviews[i],w,h = deal_text(reviews[i], font, 70)
        
        width = max(width, w)
        height = max(height, h)
        pbar.update()

        
#    width = min(width, 100)
    pbar = pyprind.ProgBar(num)
    for i in range(num):
        #print('saving pics/{}.png'.format(i))
        #print(reviews[i])
        genImage(reviews[i], width=width, height=height, font=font).save(path_data + folder +'/{}.png'.format(i))
        pbar.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Init dataset.')   
    parser.add_argument('--test', dest='test', action='store_true',
                        help='if test?')    

    args = parser.parse_args()
    if args.test:
        print("Using test mode...")
        genPics(file_name = 'minidata.csv', folder='minipics')
    else:
        genPics()
        
    
    
