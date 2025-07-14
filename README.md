# Ant-Colony-Optimisation

## Ant Colony Optimisation for Bin Packing Problem

This project implements an Ant Colony Optimisation (ACO) algorithm to solve two variations of the Bin Packing Problem (BPP1 and BPP2), as part of the ECM3412 – Nature-Inspired Computation module at the University of Exeter.

### Overview

The goal is to distribute 500 items into a fixed number of bins (10 or 50) such that the difference between the heaviest and lightest bin is minimized. The ACO algorithm is used to explore solutions, guided by pheromone updates and evaporation dynamics.

### Features

- Full ACO implementation
- Support for customisable population size and evaporation rate
- Experimental results across multiple configurations
- Reproducibility using random seeds
- Paper-style report summarizing findings

### How to run
1. Clone the repository:
   git clone https://github.com/sharpegeorge/Ant-Colony-Optimisation.git
   cd Ant-Colony-Optimisation

2. Install dependencies:
   pip install -r requirements.txt

3. Run the project:
   python ACO BPP.py
