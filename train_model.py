from tensorflow import keras

def create_model(num_classes):
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_asl_model():
    train_dir = 'data/asl_alphabet_train/asl_alphabet_train'
    test_dir = 'data/asl_alphabet_test/asl_alphabet_test'
    
    train_datagen = keras.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    test_datagen = keras.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Custom callback to print accuracy after each epoch
    class EpochCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch + 1}/10 - Training Accuracy: {logs['accuracy']:.4f} - Validation Accuracy: {logs['val_accuracy']:.4f}")
    
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[EpochCallback()],
        verbose=1
    )
    
    model.save('asl_model.h5')
    
    # Save class labels
    import json
    with open('class_labels.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    
    print("Model saved as 'asl_model.h5'")
    return model, train_generator.class_indices

if __name__ == "__main__":
    train_asl_model()