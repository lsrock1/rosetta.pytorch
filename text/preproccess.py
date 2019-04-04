from glob import glob
import json
import imageio
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import os
import h5py
# sum of square :  17357.051933220784
# sum of :  11268.273286286252
# 6088.778646934532
# sum of square :  16640.73806490707
# sum of :  10883.027124552049
# 5757.71094035502
# sum of square :  15323.923926771799
# sum of :  9691.534689084563
# 5632.389237687235


numbers = (
        '<_>',
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '.',
        # '/',
        # '@',
        # '&',
        # ',',
        # '(',
        # ')',
        # '*',
        # "'",
        # '-',
        # ':',
        # '%',
        # '=',
        # '#',
        # '$',
        # '!',
        # '?',
        # '~',
        # '_',
        # '+',
        # '"',
        'A',
        # '\/',
        # '\\',
        # ';',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I',
        'J',
        'K',
        'L',
        'M',
        'N',
        'O',
        'P',
        'Q',
        'R',
        'S',
        'T',
        'U',
        'V',
        'W',
        'X',
        'Y',
        'Z',
        # '[',
        # ']',
        # '|',
        # '<',
        # '>',
        # '^',
        # '{',
        # '}',
        # '·'
        )

number_to_ind = dict(zip(numbers, range(len(numbers))))
tnumber = 0
sum_of_square = [0.0, 0.0, 0.0]
sum_of = [0.0, 0.0, 0.0] 


def make_cropped_data(image_path, data_index, target_index):
    global tnumber
    cropped_image = []
    labels = []

    image = imageio.imread(image_path)
    anno_path = image_path.replace('.jpg', '.json')
    with open(anno_path, 'r') as data:
        json_file = json.load(data)
        for index, shape in enumerate(json_file['shapes']):
            if shape['shape_type'] == 'rectangle':
                xmin, ymin = shape['points'][0]
                xmax, ymax = shape['points'][1]
                if image[ymin:ymax, xmin:xmax, :].shape[0] <= 0:
                    continue
                cropped_image=image[ymin:ymax, xmin:xmax, :]
                try:
                    labels = [number_to_ind[number] for number in shape['label'].upper().replace(' ', '').replace(',', '') if number in numbers]
                    
                except:
                    print(shape['label'])
                
            elif shape['shape_type'] == 'polygon':
                xmin = min([i[0] for i in shape['points']])
                ymin = min([i[1] for i in shape['points']])
                xmax = max([i[0] for i in shape['points']])
                ymax = max([i[1] for i in shape['points']])
                if image[ymin:ymax, xmin:xmax, :].shape[0] <= 0:
                    continue
                cropped_image=image[ymin:ymax, xmin:xmax, :]
                labels = [number_to_ind[number] for number in shape['label'].upper().replace(' ', '').replace(',', '') if number in numbers]
            for i in range(3):
                sum_of_square[i] += np.sum(cropped_image[:, :, i].astype(np.int32) ** 2)
                sum_of[i] += np.sum(cropped_image[:, :, i].astype(np.int32))
            tnumber += cropped_image.shape[0] * cropped_image.shape[1]
            if len(labels) > 0:
                if len(labels) < 5:
                    length = 5
                elif len(labels) < 8:
                    length = 8
                elif len(labels) < 10:
                    length = 10
                elif len(labels) < 15:
                    length = 15
                else:
                    return
                if not os.path.exists('./{}'.format(length)):
                    os.makedirs('./{}'.format(length))
                if not os.path.exists('./{}/{}'.format(length, data_index)):
                    os.makedirs('./{}/{}'.format(length, target_index))
                    os.makedirs('./{}/{}'.format(length, data_index))
                #print('./{}/'.format(data_index)+image_path.split('/')[-1].replace('.jpg', '')+'_'+str(index)+'.jpg')
                # cv2.imwrite('./{}/{}/'.format(length, data_index)+image_path.split('/')[-1].replace('.jpg', '')+'_'+str(index)+'.jpg', cropped_image)
                # with open('./{}/{}/'.format(length, target_index)+image_path.split('/')[-1].replace('.jpg', '')+'_'+str(index)+'.pkl', 'wb') as f:
                #     pickle.dump(labels, f)
            #cv2.imwrite('./' + image_path.split('/')[-1], cropped_image[-1])

    return cropped_image, labels

def dashboard_to_data(path):
    cropped_images = []
    labels = []
    image_paths = sorted(glob(path))
    print(len(image_paths))
    train = image_paths[:540]
    val = image_paths[540:]
    for image_path in tqdm(train):
        make_cropped_data(image_path, 'images', 'labels')
    for image_path in tqdm(val):
        make_cropped_data(image_path, 'images_val', 'labels_val')
    #     cropped_images += data[0]
    #     labels += data[1]
    for i in range(3):
        # print('sum of square : ', sum_of_square[i]/tnumber)
        # print('sum of : ', (sum_of[i]/tnumber)**2)
        print('var : ', sum_of_square[i]/tnumber - (sum_of[i]/tnumber)**2)
        print('mean : ', sum_of[i]/tnumber)
    # with open('./images.pkl', 'wb') as handle:
    #     pickle.dump(cropped_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('./labels.pkl', 'wb') as handle:
    #     pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

def sorie_to_data(path):
    image_paths = glob(path)
    for image_path in tqdm(image_paths):
        if '(1)' in image_path or '(2)' in image_path or '(3)' in image_path or '(4)' in image_path:
            continue
        image = imageio.imread(image_path)
        text_path = image_path.replace('jpg', 'txt')
        with open(text_path, 'r') as f:
            for index, line in enumerate(f.readlines()):
                line = line.strip()
                points = [int(coordinate) for coordinate in line.split(',')[:8]]
                points = list(zip(points[::2], points[1::2]))
                target = ''.join(line.split(',')[8:])
                sorie_crop(image, points, target, index, image_path.split('/')[-1].replace('.jpg', ''))
    # for i in range(3):
    #     print('sum of square : ', sum_of_square[i]/tnumber)
    #     print('sum of : ', (sum_of[i]/tnumber)**2)
    #     print(sum_of_square[i]/tnumber - (sum_of[i]/tnumber)**2)
    #     print('mean : ', sum_of[i]/tnumber)

def sorie_crop(image, points, target, index, name):
    global tnumber
    try:
        cropped = image[points[0][1]:points[2][1], points[0][0]:points[2][0], :]
    except:
        cropped = np.repeat(np.expand_dims(image, axis=2)[points[0][1]:points[2][1], points[0][0]:points[2][0], :], 3, axis=2)
    for i in range(3):
        sum_of_square[i] += np.sum(cropped[:, :, i].astype(np.int32) ** 2)
        sum_of[i] += np.sum(cropped[:, :, i].astype(np.int32))
        tnumber += cropped.shape[0] * cropped.shape[1]
    target = [number_to_ind[i.upper()] for i in target.replace(' ', '') if i in numbers]
    
    if len(target) > 0:
        if len(target) < 5:
            length = 5
        elif len(target) < 8:
            length = 8
        elif len(target) < 10:
            length = 10
        elif len(target) < 15:
            length = 15
        else:
            return
        if not os.path.exists('./{}'.format(length)):
            os.makedirs('./{}'.format(length))
        if not os.path.exists('./{}/images'.format(length)):
            os.makedirs('./{}/images'.format(length))
            os.makedirs('./{}/labels'.format(length))
        cv2.imwrite('./{}/images/'.format(length)+name+str(index)+'.jpg', cropped)
        with open('./{}/labels/'.format(length)+name+str(index)+'.pkl', 'wb') as f:
            pickle.dump(target, f)

def shvn_crop(h5_file):
    global tnumber
    f = h5py.File(h5_file, 'r')
    digitStructName = f['digitStruct']['name']
    digitStructBbox = f['digitStruct']['bbox']
    def getName(n):
        digitindex = digitStructName[n][0]
        return ''.join([chr(c[0]) for c in f[digitindex][()]])

    def bboxHelper(attr):
        if (len(attr) > 1):
            attr = [f[attr[()][j].item()][()][0][0] for j in range(len(attr))]
        else:
            attr = [attr[()][0][0]]
        return attr

    def getBbox(n):
        bbox = {}
        bb = digitStructBbox[n].item()
        # bbox = bboxHelper(f[bb]["label"])
        bbox['height'] = bboxHelper(f[bb]["height"])
        bbox['label'] = bboxHelper(f[bb]["label"])
        bbox['left'] = bboxHelper(f[bb]["left"])
        bbox['top'] = bboxHelper(f[bb]["top"])
        bbox['width'] = bboxHelper(f[bb]["width"])
        return bbox

    def getDigitStructure(n):
        s = getBbox(n)
        s['name'] = getName(n)
        return s

    image_dict = {}
    for i in tqdm(range(len(digitStructName))):
        file_name = getName(i)
        image_dict[file_name] = getBbox(i)
        left = int(min(image_dict[file_name]['left']))
        top = int(min(image_dict[file_name]['top']))
        left_e = int(max([sum(i) for i in zip(image_dict[file_name]['left'], image_dict[file_name]['width'])]))
        top_e = int(max([sum(i) for i in zip(image_dict[file_name]['top'], image_dict[file_name]['height'])]))
        target = [int(i+1) if int(i) != 10 else 1 for i in image_dict[file_name]['label']]
        if len(target) > 0:
            if len(target) < 5:
                length = 5
            elif len(target) < 8:
                length = 8
            elif len(target) < 10:
                length = 10
            elif len(target) < 15:
                length = 15
            else:
                return
            if not os.path.exists('./{}'.format(length)):
                os.makedirs('./{}'.format(length))
            if not os.path.exists('./{}/images'.format(length)):
                os.makedirs('./{}/images'.format(length))
                os.makedirs('./{}/labels'.format(length))
            # print('./{}/images/'.format(length)+file_name.replace('.png', '_')+str(i)+'.jpg')
            
            cropped = imageio.imread('./train/'+file_name)[top:top_e, left:left_e, ::-1]
            if 0 in cropped.shape:
                if os.path.exists('./{}/images/'.format(length)+file_name.replace('.png', '_')+str(i)+'.jpg'):
                    os.remove('./{}/images/'.format(length)+file_name.replace('.png', '_')+str(i)+'.jpg')
                if os.path.exists('./{}/images/'.format(length)+file_name.replace('.png', '_')+str(i)+'.pkl'):
                    os.remove('./{}/images/'.format(length)+file_name.replace('.png', '_')+str(i)+'.pkl')
                continue
            for i in range(3):
                sum_of_square[i] += np.sum(cropped[:, :, i].astype(np.int32) ** 2)
                sum_of[i] += np.sum(cropped[:, :, i].astype(np.int32))
                tnumber += cropped.shape[0] * cropped.shape[1]
            # cv2.imwrite('./{}/images/'.format(length)+file_name.replace('.png', '_')+str(i)+'.jpg', cropped)
            # with open('./{}/labels/'.format(length)+file_name.replace('.png', '_')+str(i)+'.pkl', 'wb') as pkl:
            #     pickle.dump(target, pkl)
    f.close()
        
shvn_crop('./train/digitStruct.mat')
# sorie_to_data('./OCRDATA/*.jpg')
dashboard_to_data('./dashboard/*.jpg')