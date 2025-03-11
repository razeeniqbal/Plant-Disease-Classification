import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from datetime import datetime

class VGG16Classifier:
    def __init__(self, train_dir='train'):
        self.train_dir = train_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join('result', 'vgg16model')
        os.makedirs(self.result_dir, exist_ok=True)
        self.num_classes = len(os.listdir(train_dir))

    def create_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x = Flatten()(base_model.output)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
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
            os.path.join(self.result_dir, f'vgg16_results_{self.timestamp}.csv'),
            index=False
        )

def main():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    classifier = VGG16Classifier()
    
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