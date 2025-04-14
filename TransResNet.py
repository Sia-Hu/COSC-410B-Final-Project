import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

dataset_path = '/home/xhu/COSC410/TrashType_Image_Dataset'
garbage_types = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

dat = []
for garbage_type in garbage_types:
    folder_path = os.path.join(dataset_path, garbage_type)
    for file in os.listdir(folder_path):
        dat.append((os.path.join(dataset_path, garbage_type, file), garbage_type))
        
df = pd.DataFrame(dat, columns=['filepath', 'label'])

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

img_size = (384, 384)
batch_size = 32

train_datagen = ImageDataGenerator(
    rotation_range=45,                  
    width_shift_range=0.15,             
    height_shift_range=0.15,            
    zoom_range=0.20,                    
    horizontal_flip=True,               
    vertical_flip=True,                 
    shear_range=0.05,                   
    brightness_range=[0.9, 1.1],        
    channel_shift_range=10,             
    fill_mode='nearest',                 
    preprocessing_function=preprocess_input  
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator_aug = train_datagen.flow_from_dataframe(
    train_set,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    test_set,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

num_classes = len(train_generator_aug.class_indices)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(384, 384, 3))

base_model.trainable = True
for layer in base_model.layers[:-30]:  # freeze all but last 30 layers
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

initial_epochs = 30
history = model.fit(train_generator_aug,
                    epochs=initial_epochs,
                    validation_data=test_generator,
                    callbacks=[reduce_lr, early_stopping])

model.save('trash_classification_model_0414.h5')

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_labels = np.argmax(predictions, axis=1)
report = classification_report(true_labels, predicted_labels, target_names=class_labels)
print("\nClassification Report:\n", report)

cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

plt.savefig("trans_confusion_matrix.png", bbox_inches='tight', dpi=300)