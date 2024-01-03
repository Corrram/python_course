import time
import random


def process_data(data):
    def normalize(numbers):
        max_val = max(numbers)
        return [x / max_val for x in numbers]

    def square(numbers):
        return [x * x for x in numbers]

    def sort_numbers(numbers):
        sorted_numbers = []
        while numbers:
            min_val = min(numbers)
            numbers.remove(min_val)
            sorted_numbers.append(min_val)
        return sorted_numbers

    data = normalize(data)
    data = square(data)
    data = sort_numbers(data)
    return data


if __name__ == "__main__":
    my_data = [random.randint(1, 100) for _ in range(30_000)]
    start = time.time()
    processed_data = process_data(my_data)
    end = time.time()
    print(f"Processing took {end - start} seconds")
    print("Done!")
