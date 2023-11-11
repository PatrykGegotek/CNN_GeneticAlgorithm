import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from DataLoader.Data_Loader import load_and_scale_images
import tensorflow as tf

# Definicja podstawowej architektury sieci
def create_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(48, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


# Inicjalizacja populacji
def initialize_population(size):
    return [create_cnn() for _ in range(size)]


# Funkcja oceny sieci z użyciem generatora danych
def evaluate_networks_with_generator(networks, data_generator, num_samples):
    scores = []
    for network in networks:
        # Wczytanie wszystkich danych do tablicy
        x_all, y_all = next(data_generator, num_samples)

        # Predykcja na całym zbiorze danych
        predictions = network.predict(x_all)
        predictions = [1 if p > 0.5 else 0 for p in predictions.flatten()]

        # Obliczenie dokładności
        accuracy = np.mean(predictions == y_all)
        scores.append(accuracy)

    return scores


# Selekcja najlepszych sieci
def select_best(networks, scores, num_best):
    # Połączenie sieci z ich wynikami
    scored_networks = list(zip(networks, scores))

    # Sortowanie sieci na podstawie ich wyników
    scored_networks.sort(key=lambda x: x[1], reverse=True)  # Sortujemy malejąco, większy wynik = lepsza sieć

    # Wybór najlepszych sieci
    selected_networks = [network for network, score in scored_networks[:num_best]]

    return selected_networks


# Mutacja sieci
def mutate(network, mutation_rate=0.01, mutation_amount=0.1):
    # Iteracja przez każdą warstwę w modelu
    for layer in network.layers:
        weights = layer.get_weights()
        new_weights = []
        for weight in weights:
            # Dodajemy mutację tylko do właściwych wag, nie do biasów
            if len(weight.shape) > 1:
                mutation_mask = np.random.rand(*weight.shape) < mutation_rate
                random_mutation = np.random.normal(scale=mutation_amount, size=weight.shape)
                weight += mutation_mask * random_mutation
            new_weights.append(weight)
        layer.set_weights(new_weights)
    return network

# Krzyżowanie dwóch sieci
def crossover(parent1, parent2):
    child=create_cnn()
    for i in range(len(parent1.layers)):
        if not parent1.layers[i].get_weights():
            continue
        # Maski dla wag właściwych 
        mask0=np.random.random(np.shape(parent1.layers[i].get_weights()[0])) < 0.5
        mask0_neg = np.logical_not(mask0)
        # Maski dla biasu 
        mask1=np.random.random(np.shape(parent1.layers[i].get_weights()[1])) < 0.5
        mask1_neg = np.logical_not(mask1)
        # Ustawienie wag
        child.layers[i].set_weights([mask0*parent1.layers[i].get_weights()[0]+mask0_neg*parent2.layers[i].get_weights()[0], # wagi właściwe
                                     mask1*parent1.layers[i].get_weights()[1]+mask1_neg*parent2.layers[i].get_weights()[1]]) # bias
    return child


def genetic_algorithm(data_generator, generations, population_size, num_best):
    # Inicjalizacja populacji
    population = initialize_population(population_size)

    for generation in range(generations):
        # Ocena każdej sieci w populacji na podstawie danych z generatora
        scores = evaluate_networks_with_generator(population, data_generator, 4000)

        # Wybór najlepszych sieci
        best_networks = select_best(population, scores, num_best)

        # Tworzenie nowej populacji
        new_population = []

        # Mutacja najlepszych sieci do nowej populacji
        for network in best_networks:
            new_population.append(mutate(network))

        # Uzupełnienie nowej populacji losowo zmutowanymi sieciami
        while len(new_population) < population_size:
            random_best = np.random.choice(best_networks)
            new_population.append(mutate(random_best))

        # Aktualizacja populacji na nową generację
        population = new_population

        print(f"Generation {generation + 1}/{generations} completed.")

    # Zwracanie najlepszej sieci z ostatniej generacji
    final_scores = evaluate_networks_with_generator(population, data_generator, 4000)
    best_network = select_best(population, final_scores, 1)[0]

    return best_network


if __name__ == "__main__":
    # Uruchomienie algorytmu
    data_generator = load_and_scale_images('Data')
    best_network = genetic_algorithm(data_generator, 10, 20, 5)
