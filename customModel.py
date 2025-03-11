import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from datetime import datetime

class CustomClassifier:
    def __init__(self, train_dir='train'):
        self.train_dir = train_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join('result', 'custommodel')
        os.makedirs(self.result_dir, exist_ok=True)
        self.num_classes = len(os.listdir(train_dir))

    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_models(self, train_generator, validation_generator):
        results = []
        
        for phase in range(1, 6):
            print(f"Training Phase {phase}")
            model = self.create_model()
            
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=15
            )
            
            results.append({
                'phase': phase,
                'train_accuracy': history.history['accuracy'][-1],
                'val_accuracy': history.history['val_accuracy'][-1],
                'train_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1]
            })
        
        pd.DataFrame(results).to_csv(
            os.path.join(self.result_dir, f'custom_results_{self.timestamp}.csv'),
            index=False
        )

def main():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    classifier = CustomClassifier()
    
    train_generator = datagen.flow_from_directory(
        'train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        'train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    classifier.train_models(train_generator, validation_generator)

if __name__ == "__main__":
    main()