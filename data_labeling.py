import os
import json
import shutil
from tqdm import tqdm

# json_path = "/media/hail/HDD/DataSets/NIH_Chest_Xray/train_labels.json"
json_path = "/media/hail/HDD/DataSets/NIH_Chest_Xray/test_labels.json"
# image_folder = "/media/hail/HDD/DataSets/NIH_Chest_Xray/train"
image_folder = "/media/hail/HDD/DataSets/NIH_Chest_Xray/test"
# output_folder = "/media/hail/HDD/DataSets/NIH_Chest_Xray/labeling_image/train/"
output_folder = "/media/hail/HDD/DataSets/NIH_Chest_Xray/labeling_image/test/"

with open(json_path, 'r') as f:
    label_data = json.load(f)

disease_names = [
    "Pneumonia", "Nodule", "Mass", "Infiltration", "Pneumothorax",
    "Edema", "Pleural_Thickening", "Fibrosis", "Effusion",
    "Consolidation", "Cardiomegaly", "Atelectasis", "Hernia", "Emphysema"
]

no_finding_folder = os.path.join(output_folder, "No_Finding")
os.makedirs(no_finding_folder, exist_ok=True)

complication_folder = os.path.join(output_folder, "Complications")
os.makedirs(complication_folder, exist_ok=True)

for img_name, labels in tqdm(label_data.items(), desc="Processing Images"):
    img_path = os.path.join(image_folder, img_name)

    active_labels = [i for i, label in enumerate(labels) if label == 1]

    if len(active_labels) == 0:
        dest_path = os.path.join(no_finding_folder, img_name)
        if os.path.exists(img_path):
            shutil.copy(img_path, dest_path)
    elif len(active_labels) > 1:
        dest_path = os.path.join(complication_folder, img_name)
        if os.path.exists(img_path):
            shutil.copy(img_path, dest_path)
    else:
        for i in active_labels:
            label_name = disease_names[i]
            label_folder = os.path.join(output_folder, label_name)
            os.makedirs(label_folder, exist_ok=True)

            dest_path = os.path.join(label_folder, img_name)
            if os.path.exists(img_path):
                shutil.copy(img_path, dest_path)
