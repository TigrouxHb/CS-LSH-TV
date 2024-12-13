# CS-LSH-TV
Scalable Product Duplicate Detection Using
Locality Sensitive Hashing
This project implements a scalable method for detecting duplicate products in large e-commerce catalogs using Locality Sensitive Hashing (LSH). By leveraging LSH, the system efficiently pre-selects candidate duplicates, significantly reducing the number of necessary comparisons. The approach enhances product representations by combining title-based model words with numeric tokens from product attributes and applies hierarchical clustering for final duplicate detection.

The dataset to be used is available from: [https://personal.eur.nl/frasincar/datasets/TVs-allmerged.zip](https://personal.eur.nl/frasincar/datasets/TVs-all-merged.zip)
 as a JSON file that contains 1,624 descriptions of televisions coming from four Web
shops: Amazon.com [1], Newegg.com [2], Best-Buy.com [3], and TheNerds.net [4].

Features

Data Cleaning: Standardizes text and extracts relevant tokens.
LSH Implementation: Generates MinHash signatures to identify candidate pairs.
Clustering: Uses hierarchical clustering to finalize duplicate groups.
Evaluation: Measures performance using Pair Quality, Pair Completeness, and F1 scores.
Usage

Configure Parameters: Update the JSON_PATH in the script to point to your dataset.
Run the Script:
View Results: Outputs are saved as CSV and Excel files, and evaluation plots are displayed.
