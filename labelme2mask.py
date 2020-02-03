from __future__ import print_function

import argparse
import json
import os.path as osp
import numpy as np
import PIL.Image
import labelme

def parse_args():
    parser = argparse.ArgumentParser(description='Labelme2Mask')
    parser.add_argument('--folder_in', default='imgs', type=str,
                        help='output label file')
    parser.add_argument('--input_json', default='t001', type=str,
                        help='input image file')
    parser.add_argument('--labels_file', default='labels.txt', type=str,
                        help='output label file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    label_file = args.input_json
    folder_in = args.folder_in
    label_file = osp.join(folder_in, label_file + '.json')
    #output_mask = args.output_mask
    output_mask = osp.splitext(label_file)[0] + '_mask.png'
    #colormap = labelme.utils.label_colormap(2)

    # get class-color projections
    # (from original labelme2voc.py)
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels_file).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    
    with open(label_file) as f:
        data = json.load(f)

        img_file = osp.join(osp.dirname(label_file), data['imagePath'])
        img = np.asarray(PIL.Image.open(img_file))
        
        # save the label png file
        lbl = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data['shapes'],
            label_name_to_value=class_name_to_id,
        )
        labelme.utils.lblsave(output_mask, lbl)

if __name__ == '__main__':
    main()
