import numpy as np
import png
import pydicom
import cv2
import matplotlib.pylab as plt
import os
import shutil
import random
validation_dir = 'D:\\game\\Processed_Dataset\\validation\\images'
source_folder = r'D:\game\manifest-1600709154662\LIDC-IDRI'
output_path = r'D:\game\Processed_Dataset\training'
training_path = r'D:\game\Processed_Dataset\training'


def plot_png_with_contours(png_file, contours):
    image = cv2.imread(png_file)
    red = (255, 0, 0)
    line_width = 2
    closed = True
    annotated_image = cv2.polylines(image, [contours], closed, red, line_width)
    plt.imshow(annotated_image)
    plt.show()


def generate_mask_from_contours(contours_file, output_dir):
    contours = []
    with open(contours_file) as file:
        for line in file:
            points = line.strip().split(" ")
            contours.append([int(float(points[0])), int(float(points[1]))])
    mask = np.zeros((256, 256, 1), dtype="uint8")
    contour_points = np.array(contours).reshape((-1, 1, 2))
    mask = cv2.fillPoly(mask, [contour_points], 255)
    mask_filename = os.path.join(output_dir, "mask/img", f"{len(os.listdir(os.path.join(output_dir, 'mask/img'))) + 1}.png")
    cv2.imwrite(mask_filename, mask)


def convert_dicom_to_png(dicom_filepath, output_dir):
    dicom_data = pydicom.dcmread(dicom_filepath)
    shape = dicom_data.pixel_array.shape
    image_2d = dicom_data.pixel_array.astype(float)
    image_2d_scaled = (np.clip(image_2d, 0, None) / image_2d.max()) * 255.0
    image_2d_uint8 = np.uint8(image_2d_scaled)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    image_number = len(os.listdir(image_dir)) + 1
    png_filepath = os.path.join(image_dir, f"img{image_number}.png")
    with open(png_filepath, 'wb') as file:
        writer = png.Writer(shape[1], shape[0], greyscale=True)
        writer.write(file, image_2d_uint8)


def process_dataset_folders():
    folder_count = 0
    base_path = r'D:\game\manifest-1600709154662\LIDC-IDRI'
    for folder in os.listdir(base_path):
        if "Contours" in folder:
            contour_folder = os.path.join(base_path, folder, os.listdir(os.path.join(base_path, folder))[0])
            for subfolder in os.listdir(contour_folder):
                if "Contours" in subfolder:
                    contour_subfolder = os.path.join(contour_folder, subfolder)
                    print(folder_count, contour_subfolder)
                    for file in os.listdir(contour_subfolder):
                        if file.startswith("SC"):
                            scan_folder = os.path.join(contour_subfolder, file)
                            all_contours = os.listdir(os.path.join(scan_folder, "contours-manual/IRCCI-expert"))
                            dicom_folder = scan_folder.replace("Contours", "DICOM") + "/DICOM"
                            for contour_file in all_contours:
                                if "icontour" in contour_file:
                                    dicom_file = os.path.join(dicom_folder, f"{contour_file[:12]}.dcm")
                                    contour_file_path = os.path.join(scan_folder, "contours-manual/IRCCI-expert", contour_file)
                                    dataset_type = r'D:\game\Processed_Dataset/training/' if folder_count % 2 == 0 else r'D:\game\Processed_Dataset/testing/'
                                    generate_mask_from_contours(contour_file_path, dataset_type)
                                    convert_dicom_to_png(dicom_file, dataset_type)
                    folder_count += 1


training_path = r'D:\game\Processed_Dataset\training'


def apply_rotation(image, angle):
    height, width = image.shape[:2]
    pivot_point = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(pivot_point, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def apply_flip(image, flip_code):
    return cv2.flip(image, flip_code)


def enhance_training_data():
    image_dir = os.path.join(training_path, 'images')
    image_files = os.listdir(image_dir)
    for image_name in image_files:
        original_image = cv2.imread(os.path.join(image_dir, image_name))
        mask_image = cv2.imread(os.path.join(training_path, 'mask', image_name))
        for rotation_angle in [45, 90, 135, 180, 225, 270, 315, 0, 1]:
            new_filename = f"{image_name.split('.')[0]}_{rotation_angle}.png"
            if rotation_angle < 45:
                modified_image = apply_flip(original_image, rotation_angle)
                modified_mask = apply_flip(mask_image, rotation_angle)
            else:
                modified_image = apply_rotation(original_image, rotation_angle)
                modified_mask = apply_rotation(mask_image, rotation_angle)
            cv2.imwrite(os.path.join(image_dir, new_filename), modified_image)
            cv2.imwrite(os.path.join(training_path, 'mask', new_filename), modified_mask)


def setup_validation_dataset():
    image_dir = os.path.join(training_path, 'images')
    image_list = os.listdir(image_dir)
    if len(image_list) < 500:
        print("Insufficient images for validation.")
        return
    validation_samples = random.sample(image_list, 500)
    validation_image_dir = 'Processed_Dataset/validation/images'
    validation_mask_dir = 'Processed_Dataset/validation/mask'
    os.makedirs(validation_image_dir, exist_ok=True)
    os.makedirs(validation_mask_dir, exist_ok=True)
    for image_name in validation_samples:
        shutil.move(os.path.join(image_dir, image_name), os.path.join(validation_image_dir, image_name))
        shutil.move(os.path.join(training_path, 'mask', image_name), os.path.join(validation_mask_dir, image_name))


process_dataset_folders()
enhance_training_data()
setup_validation_dataset()

