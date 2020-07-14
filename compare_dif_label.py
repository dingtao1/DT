import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--label_path", default="/home/user1/workspace/ubuntu_autodetect_client/label.txt", type=str)
parser.add_argument("--Aresults_path", default="/home/user1/workspace/ubuntu_autodetect_client/results_caffe.txt", type=str)
parser.add_argument("--Bresults_path", default="/home/user1/workspace/ubuntu_autodetect_client/results.txt", type=str)
args = parser.parse_args()

def compare():
    A_file_path = args.label_path
    B_file_path = args.Aresults_path
    C_file_path = args.Bresults_path
    map = {}
    map2 = {}
    num = 0
    A_file = open(A_file_path, "r").readlines()
    for i in A_file:
        image_name, image_label = i.split(' ')
        map[image_name] = image_label

    C_file = open(C_file_path, "r").readlines()
    for i in C_file:
        image_name, image_label = i.split(' ')
        map2[image_name] = image_label


    B_file = open(B_file_path, "r").readlines()

    for i in B_file:
        image_name, image_label = i.split(' ')

        if map[image_name] == image_label and map2[image_name] == map[image_name]:
            continue
        if map[image_name] == image_label and map2[image_name] != map[image_name]:
            print(image_name)


if __name__ == "__main__":
    compare()