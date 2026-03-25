import copy
from controller.apply_album_aug import apply_aug
from controller.get_album_bb import get_bboxes_list
import cv2
import os
import yaml


with open("contants.yaml", 'r') as stream:
    CONSTANTS = yaml.safe_load(stream)


def run_pipeline():
    imgs = os.listdir(CONSTANTS["inp_img_pth"])
    #print(imgs)
    count=0   
    for img_file in imgs:
        print(count)
        count+=1   
        # 10_jpeg.rf.330ab36b127e02b31ce6f56bddcfa5be
        print(img_file)
        file_name = '.'.join(img_file.split('.')[:-1])
        # file_name = copy.deepcopy(img_file)
        aug_file_name = file_name + "_" + CONSTANTS["transformed_file_name"]
        image = cv2.imread(os.path.join(CONSTANTS["inp_img_pth"], img_file))           
        lab_pth = os.path.join(CONSTANTS["inp_lab_pth"], file_name + '.txt')                                
        album_bboxes = get_bboxes_list(lab_pth, CONSTANTS['CLASSES'])
        apply_aug(image, album_bboxes, CONSTANTS["out_lab_pth"],  CONSTANTS["out_img_pth"], aug_file_name, CONSTANTS["CLASSES"] , boboxespath=CONSTANTS["out_sample_pth"]
)
