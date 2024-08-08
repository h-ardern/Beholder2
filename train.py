import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
import os
import cv2
import numpy as np

# Define paths to the dataset
wider_face_train_images_dir = 'WIDER_train/images'
wider_face_train_annotations = 'wider_face_split/wider_face_train_bbx_gt.txt'


# Function to parse the annotation file
def parse_wider_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    data = []
    i = 0
    while i < len(lines):
        image_path = lines[i].strip()
        num_faces = int(lines[i + 1].strip())
        boxes = []
        for j in range(num_faces):
            box_info = list(map(int, lines[i + 2 + j].strip().split()[:4]))
            boxes.append(box_info)
        data.append((image_path, boxes))
        i += 2 + num_faces
    return data


# Parse annotations
annotations = parse_wider_annotations(wider_face_train_annotations)


# Data generator
def data_generator(annotations, image_dir, batch_size=32):
    while True:
        batch_images = []
        batch_boxes = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(annotations))
            image_path, boxes = annotations[idx]
            image = cv2.imread(os.path.join(image_dir, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128))
            boxes = np.array(boxes) / [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]  # Normalize
            batch_images.append(image)
            batch_boxes.append(boxes)

        yield np.array(batch_images), np.array(batch_boxes)


# Define the DenseNet-based face detection model
def create_densenet_model(input_shape=(128, 128, 3)):
    base_model = applications.DenseNet121(include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Output layer with 4 neurons for bounding box coordinates (x, y, width, height)
    # Using sigmoid activation to ensure the outputs are in the range [0, 1]
    output = layers.Dense(4, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['accuracy'])
    return model


model = create_densenet_model()

# Training
batch_size = 32
steps_per_epoch = len(annotations) // batch_size
train_generator = data_generator(annotations, wider_face_train_images_dir, batch_size=batch_size)

model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=10)

# Save the model
model.save('densenet_face_detection_model.h5')