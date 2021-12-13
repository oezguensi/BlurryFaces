# author: Asmaa Mirkhan ~ 2019

import os
import argparse
from glob import glob

import cv2
from DetectorAPI import Detector
import matplotlib.pyplot as plt


def blurBoxes(image, boxes, kernel_size=5):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys

    Returns:
    image -- the blurred image as a matrix
    """
    
    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]
        
        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]
        
        # apply GaussianBlur on cropped area
        blur = cv2.blur(sub, ksize=(kernel_size, kernel_size))
        
        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur
    
    return image


def main(args):
    # assign model path and threshold
    model_path = args.model_path
    threshold = args.threshold
    
    # create detection object
    detector = Detector(model_path=model_path, name="detection")
    
    # open image
    for img_path in [i for s in [glob(os.path.join(args.images_dir, f'*.{img_type}')) for img_type in args.image_types] for i in s]:
        image = cv2.imread(img_path)
        
        # real face detection
        faces = detector.detect_objects(image, threshold=threshold)
        
        # apply blurring
        image = blurBoxes(image, faces, kernel_size=args.kernel_size)
        
        # if image will be saved then save it
        if args.output_dir is not None:
            cv2.imwrite(os.path.join(args.output_dir, f'blurred_{os.path.basename(img_path)}'), image)
            print('Image has been saved successfully at', args.output_dir, 'path')
        else:
            cv2.imshow('blurred', image)
            # when any key has been pressed then close window and stop the program
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # creating argument parser
    parser = argparse.ArgumentParser(description='Image blurring parameters')
    
    # adding arguments
    parser.add_argument('-i',
                        '--images_dir',
                        help='Path to your directory containing image',
                        type=str,
                        default='../assets')
    parser.add_argument('-m',
                        '--model_path',
                        default='../face_model/face.pb',
                        help='Path to .pb model',
                        type=str)
    parser.add_argument('-o',
                        '--output_dir',
                        help='Output file path',
                        default='../outputs',
                        type=str)
    parser.add_argument('-t',
                        '--threshold',
                        help='Face detection confidence',
                        default=0.7,
                        type=float)
    parser.add_argument('-k',
                        '--kernel_size',
                        help='Face detection confidence',
                        default=30,
                        type=int)
    parser.add_argument('-s',
                        '--image_types',
                        nargs='+',
                        default=['jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'png'],
                        help='Image types to run through algorithm',
                        type=str)
    
    args = parser.parse_args()
    print(args)
    # if input image path is invalid then stop
    assert os.path.isdir(args.images_dir), 'Invalid input file'
    
    # if output directory is invalid then stop
    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)
