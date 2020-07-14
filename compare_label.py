import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--label_path", default="/home/user1/workspace/ubuntu_autodetect_client/label.txt", type=str)
parser.add_argument("--results_path", default="/home/user1/workspace/ubuntu_autodetect_client/results.txt", type=str)

args = parser.parse_args()

def compare():
    A_file_path = args.label_path
    B_file_path = args.results_path
    map = {}
    num = 0
    A_file = open(A_file_path, "r").readlines()
    for i in A_file:
        image_name, image_label = i.split(' ')
        map[image_name] = image_label
    B_file = open(B_file_path, "r").readlines()
    if(len(B_file) == 0):
        print("error: empty results.txt")
        exit(0)
    for i in B_file:
        image_name, image_label = i.split(' ')
        if map[image_name] == image_label:
            num = num + 1
    print("acc: ", num/len(B_file)*100, "%")

if __name__ == "__main__":
    compare()