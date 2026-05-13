# Gaussian Mixture Model Clustering for Jovian Atmospheric Structure

## Description

This repository contains tools to run unsupervised atmospheric clustering on sets of radiances from Jupiter's atmosphere. It also contains tools for model selection and evaluation and processes to output maps and plots of the clusters.

## Scientific Motivation

This project is in support of research by @smhill001 to characterize and analyze the physical properties of Jupiter's atmosphere near the equatorial zone. We aim to compare these clusters to human-identified regions of interests to validate the methods of traditional perception and identification of Jovian features. This comparison will inform us on how well a Gaussian Mixture Model fits Jovian atmospheric structure and possibly find new atmospheric structure that can tell us about the underlying physics. 

## Repository Structure

- data
    - HST               # Hubble Data
        - 20251016UTa   # one observation collection
            - Sys1      # data in various longitudinal mapping systems
            - Sys3  
- scripts
    - processes         # contains functions to run clustering and visualization pipeline
    - cluster_stats     
    - preprocessing
        - preprocessing
    - clustering
        - BIC
        - cluster_evaluation
        - clusters
        - gmm_distance
        - grid_search
        - silhouette
    - plots
        - plots
        - mapping

## Methodology

### Data

The input data consists of flattened and normalized radiance arrays in several wavelengths, as well as for Cloud Pressure(mb) , Ammonia Content (ppm), and AOI/CI indices. 

### Clustering

Scikit-learn was used to fit a GMM model to the data. Data was first normalized using standard scaler

### Model selection and evaluation

Models were selected by generating BIC, silhouette, and JS distance plots for various numbers of clustering components. 