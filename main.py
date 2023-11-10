import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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


# Ocena sieci
def evaluate_networks(networks, x_test, y_test):
    scores = []
    for network in networks:
        # Zakładając, że x_test i y_test są już odpowiednio przetworzone
        predictions = network.predict(x_test)
        # Prognozy mogą być w formacie [0.9, 0.2, 0.7, ...], więc konwertujemy je na etykiety klas
        predictions = [1 if p > 0.5 else 0 for p in predictions]

        # Obliczenie dokładności - procent poprawnych przewidywań
        accuracy = sum(predictions == y_test) / len(y_test)
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


# Główna pętla algorytmu genetycznego
def genetic_algorithm(x_test, y_test, generations, population_size, num_best):
    # Inicjalizacja populacji
    population = initialize_population(population_size)

    for generation in range(generations):
        # Ocena każdej sieci w populacji
        scores = evaluate_networks(population, x_test, y_test)

        # Wybór najlepszych sieci
        best_networks = select_best(population, scores, num_best)

        # Tworzenie nowej populacji
        new_population = []

        # Mutacja najlepszych sieci do nowej populacji
        for network in best_networks:
            new_population.append(mutate(network))

        # Uzupełnienie nowej populacji losowo zmutowanymi sieciami do osiągnięcia początkowej wielkości populacji
        while len(new_population) < population_size:
            # Można wybrać losową sieć z najlepszych i zmutować ją
            random_best = np.random.choice(best_networks)
            new_population.append(mutate(random_best))

        # Aktualizacja populacji na nową generację
        population = new_population

        print(f"Generation {generation + 1}/{generations} completed.")

    # Zwracanie najlepszej sieci z ostatniej generacji
    final_scores = evaluate_networks(population, x_test, y_test)
    best_network = select_best(population, final_scores, 1)[0]

    return best_network


# Uruchomienie algorytmu
# x_test, y_test - Twoje dane testowe
best_network = genetic_algorithm(x_test, y_test, 10, 20, 5)