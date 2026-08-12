# Gaussian Mixture Model Clustering for Jovian Atmospheric Structure

## Description

This repository contains tools to run unsupervised atmospheric clustering on sets of radiances from Jupiter's atmosphere. It also contains tools for model selection and evaluation and processes to output maps and plots of the clusters.

## Scientific Motivation

This project is in support of research by @smhill001 to characterize and analyze the physical properties of Jupiter's atmosphere near the equatorial zone. We aim to compare these clusters to human-identified regions of interests to validate the methods of traditional perception and identification of Jovian features. This comparison will inform us on how well a Gaussian Mixture Model fits Jovian atmospheric structure and possibly discern new sets of atmospheric structures.

## Repository Structure

```text
data/
└── HST/                         # Hubble data
    └── 20251016UTa/             # One observation collection
        ├── Sys1/                 # Data in longitudinal mapping system 1
        └── Sys3/                 # Data in longitudinal mapping system 3

scripts/
├── processes/                  # Functions for the clustering and visualization pipeline
    ├── cluster_all_regions                
    └── evaluate_all_regions
├── cluster_stats/
├── preprocessing/
│   └── preprocessing/
├── clustering/
│   ├── BIC/
│   ├── cluster_evaluation/
│   ├── clusters/
│   ├── gmm_distance/
│   ├── grid_search/
│   └── silhouette/
└── plots/
    ├── plots/
    └── mapping/
```

Data is arranged according to observation and mapping system in the data/ directory. All python scripts for the python are located in /scripts, which consists of the scripts to run the full clustering and evaluation pipelines in the processes/ directory, and other helper functions for clustering, plotting, and analyzing the output.

## Methodology

### Data

The input data consists of flattened and normalized radiance arrays in several wavelengths, as well as for Cloud Pressure(mb) , Ammonia Content (ppm), and AOI/CI indices. The datasets consist of collections of these radiance arrays (observations) that have keys of the form YYYYMMDDUTa-z. Each observation may have data associated with different mapping coordinates (eg. Sys1/Sys3)

### Clustering

Scikit-learn was used to fit a GMM model to the data. Data was first normalized using standard scaler. PCA was also implemented with scikit-learn.

### Model selection and evaluation

MGenerated BIC, silhouette, and JS distance plots for various numbers of clustering components were used to aid the cluster collection process.

## Full Clustering pipeline

The scripts in the processes/ directory were designed to process a csv with the following format: 

### Input Fields

| Field            | Description                                                               |
| ---------------- | ------------------------------------------------------------------------- |
| `Name`           | Unique name identifying the observation                                   |
| `Data Source`    | Name of the observation collection containing the input data.             |
| `PG Lat Rng`     | Planetographic latitude range to include in the analysis.                 |
| `Sys 1 Long Rng` | Longitude range in the Sys1 coordinate system to include in the analysis. |
| `ROI Dict`       | Dictionary defining the regions of interest. .                            |
| `ROI`            | Whether ROI-based processing is enabled for this configuration.           |
| `GMM`            | Whether Gaussian Mixture Model (GMM) clustering has been run              |
| `Notes`          | Additional notes                                                          |

### Additional Configuration

### Output

For each region within `regions.csv`, several clusterings can be run at once through the pipeline. Within the `config` directory, `dicts.py` specifies a cluster_runs list for all of the cluster analyses to run within the regions. It allows specification of keywords within the filenames that select radiances to analyze, a PCA flag indicating whether to perform PCA dimensionality reduction before the clustering, and a set of numbers of components to cluster on.

#### Example

````text
 {
       "dims": ["NH3", "PCld"],
       "PCA": False,
       "comps":  [4, 5, 6]
   },
   ```

will run ammonia abundance and cloud pressure clustering without PCA dimension reduction for 4, 5, and 6 clusters


Running the pipeline populates a nested directory structure for each region of interest and unique clustering run. The output visualizations
are stored in the deepest nested subdirectory. The output file names have the form:
{yyyy}-{mm}-{dd}-{parameters}_{min_lat}-{max_lat}_{min_lon}-{max_lon}_sys_{sysnum}_{visualiztion type}.png

```text
visualizations/
├── HST/                    # old hubble data
└── HST_new/                # new hubble data
    └── 20251016UTa-NEZ-A1
        ├── 275_395_502_619_631_645_673_727_889
        |    └── PCA
        |        ├── 4_cl
        |         |   └── mahalanobis_0.95
        |         |      └── 2025-10-15_275_395_502_619_631_645_673_727_889_0-15_255-285_4_sys_1_centroids.png
        |         └── 5_cl
        |
        |
        └── NH3_PCld

````

#### Running the clustering pipeline

The full clustering pipeline can be executed using the command:

`python3 -m scripts.processes.cluster_all_regions` executed from the root directory. The pipeline parameters for the runs can be modified in the config directory in the root folder.

Alternatively, to run the pipeline on a specific region for a specific set of clusters, run: `python3 -m script.pipeline` and specify the unique configuration in the file.

## Full Evaluation Pipeline


### Output

For each region within the input csv, the evaluation pipleine was run for between 2 and 10 clusterings. Within the `config` directory, `dicts.py` specifies an eval_runs list which details a collection of cluster evaluation configurations that the pipeline will run. An example configuration is:

```text

{
        "dims": ["NH3", "PCld"],
        "PCA": False
    },

```

This specifies running cluster evaluation for ammonia abundance and cloud pressure parameters without preliminary PCA dimension reduction.

The output files are of the form
{plot type}_plot_{parameters}.png

The output directory structure is shown below: 

```text
cluster_evaluations/
└── 20251016UTa
    └── 20251016UTa-NEZ-A1
        ├── 275_395_502_619_631_645_673_727_889
             ├──  BIC_plot_275_395_502_619_631_645_673_727_889.png
             ├── js_plot_275_395_502_619_631_645_673_727_889.png
             └── sil_plot_275_395_502_619_631_645_673_727_889.png

```

#### Running the clustering pipeline

The full evaluation pipeline can be executed using the command:

`python3 -m scripts.processes.evaluate_all_regions` executed from the root directory.
