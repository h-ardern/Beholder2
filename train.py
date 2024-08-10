import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
import os
import cv2
import numpy as np
from tqdm import tqdm


# Function to load and print ASCII banner
def print_ascii_banner(file_path):
    with open(file_path, 'r') as file:
        banner = file.read()
    print(banner)


# Define paths to the dataset and ASCII banner
wider_face_train_images_dir = 'WIDER_train/images'
wider_face_train_annotations = 'wider_face_split/wider_face_train_bbx_gt.txt'
ascii_banner_path = 'banner.txt'

# Print ASCII banner
print_ascii_banner(ascii_banner_path)


# Function to parse the annotation file
def parse_wider_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    data = []
    i = 0
    while i < len(lines):
        image_path = lines[i].strip()

        # Check if the next line is an image path (indicating a new entry) or a number (indicating the number of faces)
        try:
            num_faces = int(lines[i + 1].strip())
        except ValueError:
            # If the next line is not a number, we skip this entry
            i += 1
            continue

        boxes = []
        for j in range(num_faces):
            box_info = list(map(int, lines[i + 2 + j].strip().split()[:4]))
            boxes.append(box_info)
        data.append((image_path, boxes))
        i += 2 + num_faces
    return data


# Parse annotations
annotations = parse_wider_annotations(wider_face_train_annotations)

# Constants
MAX_BBOXES = 100  # Maximum number of bounding boxes per image


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
            original_height, original_width = image.shape[:2]
            image = cv2.resize(image, (128, 128))

            if len(boxes) == 0:
                # If no boxes, create padding directly
                boxes = np.zeros((MAX_BBOXES, 4))
            else:
                boxes = np.array(boxes) / [original_width, original_height, original_width,
                                           original_height]  # Normalize

                # Pad bounding boxes to ensure consistent shape
                if len(boxes) > MAX_BBOXES:
                    boxes = boxes[:MAX_BBOXES]
                else:
                    padding = np.zeros((MAX_BBOXES - len(boxes), 4))  # Create padding with shape (remaining_boxes, 4)
                    boxes = np.vstack([boxes, padding])

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
    output = layers.Dense(4 * MAX_BBOXES, activation='sigmoid')(
        x)  # 4 coordinates per bounding box, MAX_BBOXES bounding boxes

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    return model


model = create_densenet_model()


# Define the training step function
@tf.function
def train_step(images, boxes):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = tf.keras.losses.MeanSquaredError()(boxes, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Training with progress bar for epochs and batches
batch_size = 4
steps_per_epoch = len(annotations) // batch_size
train_generator = data_generator(annotations, wider_face_train_images_dir, batch_size=batch_size)

for epoch in range(100):
    print(f'Epoch {epoch + 1}/100')
    epoch_loss = 0
    with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}", unit="batch") as pbar:
        for step in range(steps_per_epoch):
            images, boxes = next(train_generator)
            boxes = boxes.reshape((batch_size, 4 * MAX_BBOXES))  # Flatten boxes to match model output
            loss = train_step(images, boxes)
            epoch_loss += loss
            pbar.set_postfix(loss=epoch_loss.numpy() / (step + 1))
            pbar.update(1)
    print(f"Epoch {epoch + 1} Loss: {epoch_loss / steps_per_epoch}")

# Save the model
model.save('densenet_face_detection_model.keras')