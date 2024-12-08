import time
import matplotlib.pyplot as plt
import numpy as np

# Import the main functions from the scripts
from outdated.FLAVTPM import main as FLAVTPM
from outdated.FMPAVTPMUU import main as FMPAVTPMUU
from outdated.FMPAVTPMDS import main as FMPAVTPMDS
from outdated.LAVTPMUU import main as LAVTPMUU
from outdated.MPLAVTPM_1 import main as MPLAVTPM_1
from outdated.MPLAVTPM_2 import main as MPLAVTPM_2


import_names = {
    FLAVTPM: 'FLAVTPM',
    LAVTPMUU: 'LAVTPMUU',
    MPLAVTPM_1: 'MPLAVTPM_1',
    MPLAVTPM_2: 'MPLAVTPM_2'
}

# Values of N to test
N_values = [10, 100, 1_000, 10_000, 20_000]

# Dictionary to store the results
benchmark_results = {name: [] for name in import_names.values()}

# Run benchmarks
for N in N_values:
    for script, name in import_names.items():
        times = []
        for _ in range(5):
            start_time = time.time()
            script(N)
            elapsed_time = time.time() - start_time
            print(f"{name} with N={N} took {elapsed_time:.2f} seconds")
            times.append(elapsed_time)
        average_time = np.mean(times)
        benchmark_results[name].append(average_time)

# Plotting the results
plt.figure(figsize=(10, 6))
for script, times in benchmark_results.items():
    plt.plot(N_values, times, marker='o', label=script)

plt.xlabel("N")
plt.ylabel("Time (seconds)")
plt.title("Benchmarking Results")
plt.legend()
plt.grid(True)
plt.show()
