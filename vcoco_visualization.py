from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
import os
import pickle


fnt = ImageFont.truetype("arial.ttf", 40)

def draw_box(im, bbox, color=(255, 0, 0), anno=''):
    draw = ImageDraw.Draw(im)
    draw.rectangle(bbox, outline=color)
    draw.text((bbox[0], bbox[3] + 10), anno, font=fnt, fill=(200, 0, 200))
    #im.show()


if __name__ == '__main__':
    pickle_path = sys.argv[1]
    res_path = sys.argv[2]
    img_pathpre = r'/mnt/lustre/share/DSK/datasets/mscoco2017/train2017/'
    f = open(pickle_path, 'rb')
    fp, fn = pickle.load(f)
    assert(not os.path.exists(res_path))
    os.mkdir(res_path)
    fp_path = os.path.join(res_path, 'fp')
    fn_path = os.path.join(res_path, 'fn')
    os.mkdir(fp_path)
    os.mkdir(fn_path)
    ar_dict = {}
    for ph in fp:
        img_id = ph[0]
        pbox = list(ph[1])
        rbox = list(ph[2])
        ar = ph[3]
        score = ph[4]
        imgpth = img_pathpre + format(img_id, '012d') + '.jpg'
        print(imgpth)
        im = Image.open(imgpth)
        save_s = ar_dict.get(ar, 1)
        ar_dict[ar] = save_s + 1
        save_n = ar + '_' + str(save_s) + '.jpg'
        save_p = os.path.join(fp_path, save_n)
        draw_box(im, pbox, color=(255,0,0))
        draw_box(im, rbox, color=(0,0,255), anno=f'{score:0.4f}')
        im.save(save_p)
    ar_dict = {}
    for ph in fn:
        img_id = ph[0]
        pbox = list(ph[1])
        rbox = list(ph[2])
        ar = ph[3]
        score = ph[4]
        imgpth = img_pathpre + format(img_id, '012d') + '.jpg'
        im = Image.open(imgpth)
        save_s = ar_dict.get(ar, 1)
        ar_dict[ar] = save_s + 1
        save_n = ar + '_' + str(save_s) + '.jpg'
        save_p = os.path.join(fn_path, save_n)
        draw_box(im, pbox, color=(255,0,0))
        draw_box(im, rbox, color=(0,0,255), anno=f'{score:0.4f}')
        im.save(save_p)
