# 🎵 Spotify Data Analysis
This project explores patterns and predictive relationships in a Spotify dataset using statistical analysis and machine learning techniques. All work was done in Python using:
> Pandas, NumPy, SciPy, Matplotlib, Scikit-learn

## Exploratory Analysis: Investigated relationships between song features like duration, popularity, loudness, energy, and more.
* Found a weak negative correlation between duration and popularity.
* Identified a strong positive correlation between loudness and energy.

## Statistical Testing:
* Used Mann-Whitney U tests to compare popularity across groups (explicit vs. non-explicit, major vs. minor key), revealing significant differences.

## Feature Evaluation for Popularity:
* Ran linear regressions for each of 10 features; instrumentalness had the highest (yet low) predictive power (R² ≈ 0.021).
* A multiple linear regression model slightly improved prediction (R² ≈ 0.0477).

## Dimensionality Reduction:
* Applied PCA to standardize and reduce feature space.
* Identified 3 meaningful principal components explaining ~57.4% of the variance.

## Classification Models:
Used logistic regression to predict song attributes:
* Key (major/minor): Speechiness was the best predictor (AUROC ≈ 0.56).
* Genre (classical or not): PCA-transformed features outperformed single-feature models (AUROC ≈ 0.95).
* Explicitness: Explored energy and loudness as predictors; loudness was stronger but still modest.

## Interesting Insight
Louder, high-energy tracks tend to be more likely labeled as explicit, aligning with real-world music trends, though neither is a strong standalone predictor.
