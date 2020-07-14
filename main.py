import cv2
import random
import os
import shutil
from PIL import Image
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--bmp", default=False)
parser.add_argument("--num", default=5, type=int)
parser.add_argument("--input_dir", default="/home/user1/workspace/imagnet/ILSVRC2012_img_val/", type=str)
parser.add_argument("--input_txt", default="/home/user1/workspace/imagnet/caffe_ilsvrc12/val.txt", type=str)
parser.add_argument("--output_dir", default="/home/user1/workspace/imagnet/", type=str)

args = parser.parse_args()




def imagenet_rand(num):
    input_dir_path = args.input_dir
    output_dir_path = args.output_dir
    input_txt_path = args.input_txt
    input_txt = open(input_txt_path, "r")

    images = input_txt.readlines()
    images = random.sample(images, num)

    output_dir_path = output_dir_path + "imagenet_random"+str(args.num)+"/"
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    output_txt_path = output_dir_path + "label.txt"
    output_txt = open(output_txt_path, "w")
    for image in images:
        if args.bmp == True:
            image, label = image.split(' ')
            image = image.split('.')[0] + ".bmp" + ' ' + label
            output_txt.write(image)
        else:
            output_txt.write(image)
    output_images_dir = output_dir_path + "images/"
    if not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)
    list_file = os.listdir(output_images_dir)
    for f in list_file:
        if os.path.isfile(output_images_dir+f) == True:
            os.remove(output_images_dir+f)
    for image in images:
        if args.bmp == True:
            image_name = image.split(' ')[0]
            image_path = input_dir_path + image_name
            im = Image.open(image_path)
            im.save(output_images_dir+image_name.split(".")[0]+".bmp")
        else:
            image_name = image.split(' ')[0]
            shutil.copyfile(input_dir_path + image_name, output_images_dir + image_name)

if __name__ == "__main__":
    imagenet_rand(args.num)