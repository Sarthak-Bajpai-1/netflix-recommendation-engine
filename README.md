# Netflix-Style Movie Recommendation Engine

A movie recommendation engine built on the Netflix Prize dataset, developed as a capstone project for the
Advanced Data Scientist Program (IIT Roorkee / iHUB DivyaSampark, in collaboration with Intellipaat).

## Problem Statement

Customer behaviour prediction sits at the core of most modern business models. OTT platforms like Netflix and
Amazon analyze user activity patterns to recommend content suited to individual preferences, increasing user
engagement. This project builds a recommendation engine from the ground up: given a user's rating history, the
system recommends movies best suited to them.

## Dataset

This project uses the [Netflix Prize dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data),
specifically:
- `combined_data_1.txt` — ~24 million customer ratings across 4,499 movies
- `movie_titles.csv` — movie ID, release year, and title lookup

The raw ratings file is **not included in this repo** due to size (~495MB, exceeding GitHub's file size limit).
See [Setup](#setup) below for how to get it.

## Approach

1. **Parsing** — `combined_data_1.txt` uses a non-tabular format (movie ID header lines followed by rating
   rows). Parsed using a vectorized pandas technique rather than a row-by-row loop, for speed and memory
   efficiency on the full 24M-row file.
2. **Exploratory analysis** — most popular movies by rating volume, highest/lowest rated movies subject to a
   minimum rating-volume threshold.
3. **Recommendation model** — the full customer x movie matrix is too large and sparse to work with directly,
   so it's trimmed using benchmark-quantile filtering (keeping movies/customers with enough ratings for
   reliable correlations). An item-based collaborative filtering approach (Pearson correlation between movie
   rating vectors) is then used to recommend unseen movies similar to a customer's highest-rated title, with a
   popularity-based fallback for customers outside the trimmed matrix.

## Repo structure

```
notebooks/   Jupyter notebook with the full pipeline
data/        movie_titles.csv (ratings file excluded — see Setup)
outputs/     generated charts and recommendation CSV
```

## Setup

```bash
git clone https://github.com/<your-username>/netflix-recommendation-engine.git
cd netflix-recommendation-engine
pip install -r requirements.txt
```

Download `combined_data_1.txt` from the
[Netflix Prize dataset on Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and place it
in the `data/` folder, then run the notebook in `notebooks/`.

## Results

- Identified the most-rated and highest/lowest-rated titles across the dataset.
- Built a working per-customer recommendation pipeline, generating recommendations for ~24,000 customers.
- Outputs saved to `outputs/customer_recommendations.csv`.

## Possible extensions

- Run across all four `combined_data_*.txt` files instead of one.
- Compare against a matrix factorization baseline (`TruncatedSVD`, or the `surprise` library) evaluated with
  RMSE on a held-out test split.
- Incorporate content-based features (cast, description) alongside collaborative filtering.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter
