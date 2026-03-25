import cv2
from controller.album_to_yolo_bb import multi_obj_bb_yolo_conversion
from controller.album_to_yolo_bb import single_obj_bb_yolo_conversion
from controller.save_augs import save_aug_image, save_aug_lab
from controller.validate_results import draw_yolo
import albumentations as A
import uuid
import os


transforms = [
    # Pipeline 1
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RandomRotate90(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 2
    # A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 3
    # A.Compose([
    #     A.RandomScale(scale_limit=0.2, p=0.5),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 4
    # A.Compose([
    #     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 5
    A.Compose([
        A.Blur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 6
    A.Compose([
        A.RandomSnow(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 7
    A.Compose([
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=0.2),
        A.RandomBrightnessContrast(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 8
    A.Compose([
        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                     blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=True, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 9
    A.Compose([
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5,
                       always_apply=False, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 10
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.CLAHE(clip_limit=(1, 4), p=0.3),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 11
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.MotionBlur(blur_limit=7, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 12
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RandomScale(scale_limit=0.3, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 13
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.ElasticTransform(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 14
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 15
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 16
    # A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.Blur(blur_limit=(3, 5), p=0.3),
    #     A.RandomRotate90(p=0.5),
    #     A.RandomScale(scale_limit=0.2, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 17
    # A.Compose([
    #     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    #     A.GaussianBlur(blur_limit=(5, 7), p=0.3),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 18
    # A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.GridDistortion(p=0.3),
    #     A.RandomScale(scale_limit=0.2, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 19
    # A.Compose([
    #     A.Blur(blur_limit=(3, 5), p=0.3),
    #     A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 20
    # A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.Blur(blur_limit=(3, 5), p=0.3),
    #     A.RandomRotate90(p=0.5),
    #     A.RandomScale(scale_limit=0.2, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 21
    # A.Compose([
    #     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    #     A.GaussianBlur(blur_limit=(5, 7), p=0.3),
    #     A.HorizontalFlip(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 22
    # A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.GridDistortion(p=0.3),
    #     A.RandomScale(scale_limit=0.2, p=0.5),
    #     A.RandomRotate90(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 23
    # A.Compose([
    #     A.Blur(blur_limit=(3, 5), p=0.3),
    #     A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    #     A.GaussNoise(var_limit=(10, 50), p=0.3),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 24
    # # A.Compose([
    # #     A.HorizontalFlip(p=0.5),
    # #     A.GridDistortion(p=0.3),
    # #     A.RandomScale(scale_limit=0.2, p=0.5),
    # #     A.RandomRotate90(p=0.5),
    # #     A.RandomBrightnessContrast(p=0.5),
    # # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 25
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    #     A.RandomScale(scale_limit=0.3, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 26
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    #     A.ElasticTransform(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 27
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 28
    # # A.Compose([
    # #     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    # #     A.GaussianBlur(blur_limit=(5, 7), p=0.3),
    # #     A.RandomBrightnessContrast(p=0.5),
    # #     A.HorizontalFlip(p=0.5),
    # # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 29
    # # A.Compose([
    # #     A.HorizontalFlip(p=0.5),
    # #     A.GridDistortion(p=0.3),
    # #     A.RandomScale(scale_limit=0.2, p=0.5),
    # #     A.RandomRotate90(p=0.5),
    # #     A.RandomBrightnessContrast(p=0.5),
    # # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 30
    # A.Compose([
    #     A.Blur(blur_limit=(3, 5), p=0.3),
    #     A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    #     A.GaussNoise(var_limit=(10, 50), p=0.3),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 31
    # A.Compose([
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 32
    # A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.Blur(blur_limit=(3, 5), p=0.3),
    #     A.RandomRotate90(p=0.5),
    #     A.RandomScale(scale_limit=0.2, p=0.5),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 33
    # A.Compose([
    #     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    #     A.GaussianBlur(blur_limit=(5, 7), p=0.3),
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 34
    # A.Compose([
    #     A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0),
    #     A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 35
    # A.Compose([
    #     A.ShiftScaleRotate(p=0.5),
    #     A.RandomBrightnessContrast(p=0.3),
    #     A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 36
    # A.Compose([
    #     A.AdvancedBlur(blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=90,
    #                 beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), always_apply=False, p=0.5),
    #     A.ShiftScaleRotate(p=0.8),
    #     A.RandomBrightnessContrast(p=0.2),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 37
    # A.Compose([
    #     A.MotionBlur(blur_limit=7, allow_shifted=True, always_apply=False, p=0.5),
    #     A.ShiftScaleRotate(p=0.8),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 38
    # A.Compose([
    #     A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5),
    #     A.ShiftScaleRotate(p=0.7),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 39
    # A.Compose([
    #     A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=False, p=0.5),
    #     A.ShiftScaleRotate(p=0.6),
    #     A.RandomBrightnessContrast(p=0.6),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 40
    # A.Compose([
    #     A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=False, p=0.5),
    #     A.ShiftScaleRotate(p=0.7),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 41
    # A.Compose([
    #     A.Posterize(num_bits=4, always_apply=False, p=0.5),
    #     A.ShiftScaleRotate(p=0.7),
    #     A.RandomBrightnessContrast(p=0.6),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 42
    # A.Compose([
    #     A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200),
    #                 blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=True, p=0.5),
    #     A.ShiftScaleRotate(p=0.7),
    #     A.RandomBrightnessContrast(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 43
    # A.Compose([
    #     A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5,
    #                 always_apply=False, p=0.5),
    #     #A.ShiftScaleRotate(p=0.5),
    #     A.RandomBrightnessContrast(p=0.6),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 44
    # A.Compose([
    #     A.CLAHE(),
    #     A.Transpose(),
    #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.20, p=.75,rotate_limit=15),
    #     A.Blur(blur_limit=3),
    #     A.OpticalDistortion(),
    #     A.GridDistortion(),
    #     A.HueSaturationValue()], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 45
    # A.Compose([
    #     A.Transpose(),
    #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.20, p=.75, rotate_limit=0),
    #     A.Blur(blur_limit=3),
    #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # Pipeline 46
    # A.Compose([
    #     A.RandomScale(scale_limit=0.4, p=0.3),
    #     A.ElasticTransform(p=0.5),
    #     A.Emboss(p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 47
    # A.Compose([
    #     A.RandomSnow(p=0.5),
    #     A.GridDistortion(p=0.3),
    #     A.ToGray(p=0.1),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 48
    # A.Compose([
    #     A.RandomScale(scale_limit=0.2, p=0.3),
    #     A.OpticalDistortion(p=0.5),
    #     A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 49
    # A.Compose([
    #     A.RandomSnow(p=0.7,brightness_coeff=0.5),
    #     A.Emboss(p=0.2),
    # ], bbox_params=A.BboxParams(format='yolo')),

    # # Pipeline 50
    # A.Compose([
    #     A.GridDistortion(p=0.3),
    #     A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, p=0.2),
    # ], bbox_params=A.BboxParams(format='yolo'))
]
#3,31,37,42,74
def apply_aug(image, bboxes, out_lab_pth, out_img_pth, transformed_file_name, classes, boboxespath):
    
    selected_indices = [0, 1, 3]
    # for idx, i in enumerate(transforms):
    for idx in selected_indices:
        print("index ",idx)
        transformed = transforms[idx](image=image, bboxes=bboxes)

        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        # print(transformed_bboxes)
        tot_objs = len(transformed_bboxes)
        #print(tot_objs)
        uniqueId = str(uuid.uuid1().hex)
        if tot_objs > 0:
            if tot_objs > 1:
                transformed_bboxes = multi_obj_bb_yolo_conversion(transformed_bboxes, classes)
                save_aug_lab(transformed_bboxes, out_lab_pth,
                             transformed_file_name + uniqueId + "index-" + str(idx) + ".txt")
            else:
                transformed_bboxes = [single_obj_bb_yolo_conversion(transformed_bboxes[0], classes)]
                save_aug_lab(transformed_bboxes, out_lab_pth,
                             transformed_file_name + uniqueId + "index-" + str(idx) + ".txt")
            save_aug_image(transformed_image, out_img_pth, transformed_file_name + uniqueId + "index-" + str(idx) + ".jpg")
            rootpath = boboxespath
            if not os.path.exists(rootpath):
                os.makedirs(rootpath)

            quickPath = os.path.join(rootpath, transformed_file_name + uniqueId + "index-" + str(idx) + ".jpg")

            draw_yolo(transformed_image, transformed_bboxes,path=quickPath)
        else:
            # print("label file is empty","index",i)
            print("label file is empty","index",idx)