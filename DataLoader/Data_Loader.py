from keras.preprocessing.image import ImageDataGenerator
import os

def load_and_scale_images(data_dir, target_size=(64, 64), batch_size=600):
    # Ustawienia generatora obrazów
    datagen = ImageDataGenerator(rescale=1./255)

    # Przygotowanie generatora danych
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',  # Używamy klasyfikacji binarnej (koty vs psy)
        classes=['cats', 'dogs']
    )

    return train_generator

# Przykład użycia:
# data_generator = load_and_scale_images('Data')
# x_train, y_train = next(data_generator)  # Wczytuje jedną partię danych