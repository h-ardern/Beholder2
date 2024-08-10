import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Function to parse the annotation file
def parse_wider_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    data = []
    i = 0
    while i < len(lines):
        image_path = lines[i].strip()

        try:
            num_faces = int(lines[i + 1].strip())
        except ValueError:
            i += 1
            continue

        boxes = []
        for j in range(num_faces):
            box_info = list(map(int, lines[i + 2 + j].strip().split()[:4]))
            boxes.append(box_info)
        data.append((image_path, boxes))
        i += 2 + num_faces
    return data


# Function to generate test data
def data_generator(annotations, image_dir, batch_size=32):
    while True:
        batch_images = []
        batch_boxes = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(annotations))
            image_path, boxes = annotations[idx]
            image = cv2.imread(os.path.join(image_dir, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image.shape[:2]
            image = cv2.resize(image, (128, 128))

            if len(boxes) == 0:
                boxes = np.zeros((MAX_BBOXES, 4))
            else:
                boxes = np.array(boxes) / [original_width, original_height, original_width, original_height]

                if len(boxes) > MAX_BBOXES:
                    boxes = boxes[:MAX_BBOXES]
                else:
                    padding = np.zeros((MAX_BBOXES - len(boxes), 4))
                    boxes = np.vstack([boxes, padding])

            batch_images.append(image)
            batch_boxes.append(boxes)

        yield np.array(batch_images), np.array(batch_boxes)


# Paths
model_path = 'densenet_face_detection_model.keras'
val_images_dir = 'WIDER_val/images'  # Update this path
val_annotations_path = 'wider_face_split/wider_face_val_bbx_gt.txt'  # Update this path

# Load the model
model = tf.keras.models.load_model(model_path)

# Parse the validation annotations
val_annotations = parse_wider_annotations(val_annotations_path)

# Constants
MAX_BBOXES = 10  # Maximum number of bounding boxes per image
batch_size = 32

# Generate validation data
val_generator = data_generator(val_annotations, val_images_dir, batch_size=batch_size)

# Validation loop
y_true = []
y_pred = []

steps = len(val_annotations) // batch_size

# Progress bar tracking the entire validation process
with tqdm(total=steps, desc="Validating", unit="batch") as pbar:
    for _ in range(steps):
        images, true_boxes = next(val_generator)
        predictions = model.predict(images, verbose=0)  # Disable verbose output from Keras

        # Reshape the predictions and true boxes
        predictions = predictions.reshape((batch_size, MAX_BBOXES, 4))

        for i in range(batch_size):
            # Flatten the boxes for comparison
            true_boxes_flat = true_boxes[i].flatten()
            predictions_flat = predictions[i].flatten()

            y_true.extend(true_boxes_flat)
            y_pred.extend(predictions_flat)

        pbar.update(1)

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Apply threshold to predictions to convert them to binary
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)

# Compute F1 score, precision, and recall
f1 = f1_score(y_true.astype(int), y_pred_binary, average='macro')
precision = precision_score(y_true.astype(int), y_pred_binary, average='macro')
recall = recall_score(y_true.astype(int), y_pred_binary, average='macro')

print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true.astype(int), y_pred_binary)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()b