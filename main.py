from seam_carving import SeamCarver
import os
import argparse
import time

def parse_args():
    
    parser = argparse.ArgumentParser(description='Seam Carving')
    
    parser.add_argument('--folder_in', default='imgs', type=str, 
                        help='input folder')
    parser.add_argument('--folder_out', default='out', type=str,
                        help='output folder')
    parser.add_argument('--new_height', default=400, type=int,
                        help='new height')
    parser.add_argument('--new_width', default=400, type=int,
                        help='new width')
    parser.add_argument('--filename_input', default='t009', type=str,
                        help='input image file')
    parser.add_argument('--object_mask', default='p009', type=str,
                        help='object mask filename')
    parser.add_argument('--protect_mask', default='t009', type=str,
                        help='pretect mask filename')
    parser.add_argument('--filename_output', default='t009', type=str,
                        help='output image file')
    parser.add_argument('--demo', default=False, type=bool,
                        help='demo is with plot')
    parser.add_argument('--mode', default='protect', type=str,
                        help='protect: without_mask; protect: protect_mask; remove: remove_object; both: protect_remove')
    parser.add_argument('--fast_mode', default=False, type=bool,
                        help='use fastmode to speed up processing')
    args = parser.parse_args()
    return args


def image_resize_without_mask(filename_input, filename_output, new_height, new_width):
    '''
    没有保护mask，直接最简单的resize
    '''
    obj = SeamCarver(filename_input, new_height, new_width, fast_mode=args.fast_mode)
    obj.save_result(filename_output)


def image_resize_with_mask(filename_input, filename_output, new_height, new_width, filename_mask):
    '''
    有一个保护的mask，可以选择主要想要保留的部分
    '''
    obj = SeamCarver(filename_input, new_height, new_width, protect_mask=filename_mask, fast_mode=args.fast_mode)
    obj.save_result(filename_output)


def object_removal(filename_input, filename_output, filename_mask, demo):
    '''
    有一个需要移除的object的mask
    '''
    obj = SeamCarver(filename_input, 0, 0, object_mask=filename_mask, demo=demo, fast_mode=args.fast_mode)
    obj.save_result(filename_output)

def protect_and_removal(filename_input, filename_output, protect_mask, object_mask, demo):
    '''
    同时清除和保留多个对象
    '''
    obj = SeamCarver(filename_input, 0, 0, object_mask=object_mask, protect_mask=protect_mask, demo=demo, fast_mode=args.fast_mode)
    obj.save_result(filename_output)

if __name__ == '__main__':
    '''
    原始图像和mask图像都放在folder_in当中
    输出图像放在folder_out中
    '''
    args = parse_args()

    folder_in = args.folder_in
    folder_out = args.folder_out
    demo = args.demo

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    filename_input = args.filename_input + '.jpg'
    filename_output = args.filename_output + '_out.png'
    protect_mask = args.protect_mask + '_mask.png'
    object_mask = args.object_mask + '_mask.png'
    #filename_mask = args.filename_mask
    new_height = args.new_height
    new_width = args.new_width
    mode = args.mode

    
    input_image = os.path.join(folder_in, filename_input)
    protect_mask = os.path.join(folder_in, protect_mask)
    object_mask = os.path.join(folder_in, object_mask)
    output_image = os.path.join(folder_out, filename_output)
    

    '''
    三个可用模式，分别代表三个功能
    '''
    start = time.time()
    if mode == 'plain':
        image_resize_without_mask(input_image, output_image, new_height, new_width)
    elif mode == 'protect':
        image_resize_with_mask(input_image, output_image, new_height, new_width, protect_mask)
    elif mode == 'remove':
        object_removal(input_image, output_image, object_mask, demo)
    elif mode == 'both':
        protect_and_removal(input_image, output_image, protect_mask, object_mask, demo)
    else:
        raise Exception("Mode Error!!")
    print(time.time()-start)



