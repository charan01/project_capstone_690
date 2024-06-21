# Generating Synthetic Flight Tracks using GANs

## Introduction
Welcome to the Synthetic Flight Tracks Generation project! This project focuses on generating synthetic flight tracks using Generative Adversarial Networks (GANs) to enhance Air-to-air collision Risk Modeling (CRM). In this README file, we'll provide an overview of the project, details about the dataset used, the problem statement, and instructions on navigating through the project's folders.

## About the Dataset
The dataset used in this project is the Zurich Runway Dataset, which captures airplane paths and their characteristics. It provides a comprehensive set of features including:

- Time-stamped events during flights from October 1st to November 30th, 2019, in UTC.
- Geographic coordinates, groundspeed, altitude, and geoaltitude offering a complete spatial picture.
- Nearly complete dataset with minimal missing values.
- 350 unique "flight_id" values, each associated with precisely 200 data records.
- Accurate latitude and longitude data depicting expected flight paths.
- Groundspeed values corresponding to normal aircraft speeds, ensuring dataset accuracy.

## Problem Statement
Current Air-to-air collision Risk Modeling (CRM) relies on historic and simulated flight tracks. However, historic tracks may not represent new technologies, and simulating enough tracks is challenging. To address this, our project aims to apply Generative Adversarial Networks (GANs) to generate over 1 million synthetic flight tracks by training neural networks on historic data.

## Project Structure
This GitHub repository is structured as follows:

- **code**: This folder contains all code files for the project, including scripts for data preprocessing, model training, and evaluation.
- **data**: Here, you can find the project dataset, the Zurich Runway Dataset, used for training and testing our models.

## Client
Our client for this project is the GMU Centre for Aviation Transportation Systems Research (CATSR), and we aim to provide them with a robust solution for generating synthetic flight tracks to enhance Air-to-air collision Risk Modeling (CRM).

## Getting Started
To explore this project and its findings, follow these steps:

1. Clone this GitHub repository to your local machine.
2. Navigate to the `data` folder to access the dataset.
3. Explore the `code` folder to review the scripts and Jupyter Notebook files detailing the data preprocessing, model training, and evaluation processes.

To run the code:

1. Start by executing the main code file `RNN_GAN.ipynb`. This notebook requires support from two additional files:
   - `model_train.py` contains the main GAN model code.
   - `utils.py` provides essential functions, including `random_generator` for generating noise.
  
## Visualizations

In the `visualizations` folder, you'll find graphical representations including:

1. Histograms depicting Along Track Distances between real and synthetic data and each point distances between synthetic and real data.   
2. Visual comparisons of real and synthetic flight tracks, presented as images generated using Tableau.

## Conclusion
The Synthetic Flight Tracks Generation project offers a promising approach to enhancing Air-to-air collision Risk Modeling (CRM) by generating synthetic flight tracks using GANs. By leveraging historic data and advanced machine learning techniques, we aim to provide valuable insights for the aviation industry.

Feel free to reach out if you have any questions or need further clarification. Thank you for exploring our project!
