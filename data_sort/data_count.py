import json
from collections import Counter

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = json.load(f)
    return labels

def count_classes(labels):
    class_counts = Counter()
    no_disease_count = 0

    for img, label in labels.items():
        if sum(label) == 0:
            no_disease_count += 1
        else:
            for i, value in enumerate(label):
                if value == 1:
                    class_counts[i] += 1

    return class_counts, no_disease_count

def main():
    train_label_file = '/media/hail/HDD/DataSets/NIH_Chest_Xray/train_labels.json'
    test_label_file = '/media/hail/HDD/DataSets/NIH_Chest_Xray/test_labels.json'

    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    train_labels = load_labels(train_label_file)
    test_labels = load_labels(test_label_file)

    train_class_counts, train_no_disease_count = count_classes(train_labels)
    test_class_counts, test_no_disease_count = count_classes(test_labels)

    print("Train Dataset Class Counts:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {train_class_counts[i]}")
    print(f"No Disease: {train_no_disease_count}")

    print("\nTest Dataset Class Counts:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {test_class_counts[i]}")
    print(f"No Disease: {test_no_disease_count}")

if __name__ == '__main__':
    main()

