# VRPTW-GA Demo

A Streamlit-based demo application for solving the Vehicle Routing Problem with Time Windows (VRPTW) using a Genetic Algorithm (GA).

## Description

This project demonstrates the application of genetic algorithms to optimize vehicle routing for package collection and delivery with time window constraints. The system handles 20 senders and 20 recipients, ensuring vehicles operate within capacity limits and respect pickup/delivery time windows.

## Features

- Interactive data editing for senders and recipients
- Genetic algorithm optimization with configurable generations
- Visualization of locations and optimized routes
- Detailed results explanation including fitness and objective values
- Time arrival tables for packages

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd VRPTW-GA
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```
   streamlit run vrptw-ga.py
   ```

2. Open your browser to the provided local URL (usually http://localhost:8501)

3. Use the interface to:
   - View and edit input data
   - Adjust simulation settings (number of generations)
   - Run the genetic algorithm
   - Explore results and visualizations

## Requirements

- Python 3.7+
- Dependencies listed in requirements.txt:
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - deap

## Project Structure

- `vrptw-ga.py`: Main application file
- `requirements.txt`: Python dependencies
- `README.md`: This file

![Demo UI](assets/demo1.jpg)

![Demo UI](assets/demo2.jpg)

![Demo UI](assets/demo3.jpg)

![Demo UI](assets/demo4.jpg)
