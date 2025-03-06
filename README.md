# ADM1 Multi-Simulation Dashboard

This Streamlit application provides a user-friendly interface for simulating the Anaerobic Digestion Model No. 1 (ADM1). It allows running **three concurrent simulations** with different reactor parameters on the same feedstock. The application integrates a GenAI assistant (powered by Google's Gemini API) to help users select appropriate state variable values based on natural language descriptions of feedstock.

## Features

- **Three Concurrent Simulations**: Run and compare up to three ADM1 simulations with different reactor parameters (temperature, HRT, integration method) on the same feedstock
- **AI Assistant**: Get recommendations for ADM1 state variables by describing your feedstock in natural language
- **Interactive Parameter Adjustment**: Fine-tune model parameters with sliders and input fields
- **Comprehensive Visualization**: Multiple plot types for comparing simulation results
- **Export Capabilities**: Export interactive plots as HTML files and simulation data as CSV for further analysis