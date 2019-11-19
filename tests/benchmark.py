"""Benchmark execution time of all of the simulators."""
import time

import numpy as np

import whynot as wn


def benchmark_dynamic_simulator(simulator, name, num_trials=10):
    """Run dynamical system simulator repeatedly and report execution time statistics."""
    timings = []
    for _ in range(num_trials):
        initial_state = simulator.State()
        config = simulator.Config()
        start_time = time.perf_counter()
        simulator.simulate(initial_state, config)
        timings.append(time.perf_counter() - start_time)
    print(
        f"{name.upper()}:\t average={np.mean(timings):.3f} s, std={np.std(timings):.3f} s, "
        f"max={np.max(timings):.3f} s, (trials={num_trials})"
    )


def benchmark_agent_based_model(simulator, name, num_agents=100, num_trials=10):
    """Run the agent-based model repeatedly and report execution time statistics."""
    timings = []
    for _ in range(num_trials):
        agents = [simulator.Agent() for _ in range(num_agents)]
        config = simulator.Config()
        start_time = time.perf_counter()
        simulator.simulate(agents, config)
        timings.append(time.perf_counter() - start_time)
    print(
        f"{name.upper()}:\t average={np.mean(timings):.3f} s, std={np.std(timings):.3f} s, "
        f"max={np.max(timings):.3f} s, (trials={num_trials})"
    )


def benchmark_all():
    """Run benchmarks for each simulator."""
    benchmark_dynamic_simulator(wn.dice, "dice")
    benchmark_dynamic_simulator(wn.hiv, "hiv")
    benchmark_dynamic_simulator(wn.lotka_volterra, "lotka-volterra")
    benchmark_dynamic_simulator(wn.opioid, "opioid")
    benchmark_dynamic_simulator(wn.world2, "world2")
    benchmark_dynamic_simulator(wn.world3, "world3")

    benchmark_agent_based_model(wn.civil_violence, "civil_violence")


if __name__ == "__main__":
    benchmark_all()
