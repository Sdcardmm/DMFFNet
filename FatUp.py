import cv2
import os
import numpy as np

input_folders = ['../data/gene/fat']
output_folders = ['../data/gene/fat_1']

for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

def enhance_fat_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val != min_val:
        stretched = (image - min_val) * (255.0 / (max_val - min_val))
    else:
        stretched = image.copy()
    stretched = stretched.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(stretched)

    return enhanced

for in_folder, out_folder in zip(input_folders, output_folders):
    for filename in os.listdir(in_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(in_folder, filename)
            if not os.path.exists(img_path):
                print(f"警告：文件 {img_path} 不存在")
                continue
            try:
                with open(img_path, 'rb') as f:
                    img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                print(f"警告：读取图像 {img_path} 时出错，错误信息：{e}")
                continue

            enhanced_img = enhance_fat_image(img)

            output_path = os.path.join(out_folder, filename)
            cv2.imwrite(output_path, enhanced_img)
            print(f"已处理并保存：{output_path}")