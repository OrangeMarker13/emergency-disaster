# Quantum Relief Router

Quantum Relief Router is a disaster-response routing application built with Streamlit, NetworkX, OpenStreetMap, and Qiskit.

The application models emergency transportation networks and uses QAOA, the Quantum Approximate Optimization Algorithm, to select efficient routes during disasters.

## What It Does

1. Accepts an emergency location.
2. Retrieves nearby road-network data.
3. Builds a NetworkX road graph.
4. Simulates disaster-related road closures.
5. Generates candidate routes.
6. Runs Dijkstra and A* as classical baselines.
7. Converts route selection into a QUBO optimization problem.
8. Uses QAOA to optimize route selection.
9. Displays routes on an interactive map.
10. Displays quantum route probabilities.
11. Compares quantum and classical routing performance.

## Technology

- Python
- Streamlit
- Qiskit
- Qiskit Aer
- Qiskit Algorithms
- NetworkX
- OpenStreetMap
- PyDeck
- NumPy
- pandas
- SciPy

## Project Structure

```text
quantum-relief-router/
│
├── app.py
├── classical_router.py
├── config.py
├── map_data.py
├── quantum_router.py
├── simulation.py
├── telemetry.py
├── visualization.py
├── requirements.txt
├── README.md
└── .gitignore
