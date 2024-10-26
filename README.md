# Suicide Detection Project

This project focuses on detecting suicidal tendencies in text data using various Natural Language Processing (NLP) techniques and machine learning models. The primary objective is to preprocess the data for cleaner representation, vectorize the processed text, and employ classifiers to predict if a text indicates suicidal thoughts.

## Project Overview

- **Dataset:** The dataset contains labeled text data with classes indicating 'suicide' or 'not suicide'.
- **Objective:** Perform extensive text preprocessing, feature extraction, and apply machine learning algorithms to classify text into respective classes.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is a crucial step in understanding the underlying patterns, trends, and distributions within the dataset. Here’s a comprehensive overview of the EDA process we followed:

1. **Data Loading and Inspection:**
   - Loaded the dataset and displayed the first few rows to understand its structure, features, and types.
   - Checked the shape of the dataset to assess the number of samples and features.

2. **Missing Values Analysis:**
   - Analyzed the dataset for missing values in each column using visualizations such as heatmaps.
   - Summarized the percentage of missing values to determine the extent of data cleaning needed.

3. **Class Distribution:**
   - Created visualizations (e.g., bar charts) to illustrate the distribution of classes (i.e., 'suicide' and 'not suicide').
   - Checked for class imbalance, which could impact model performance, and planned to implement strategies to address it if necessary.

4. **Text Length Analysis:**
   - Analyzed the distribution of text lengths in both classes using histograms to understand if one class tends to have longer or shorter texts.
   - Used boxplots to visually represent the range and central tendency of text lengths.

5. **N-gram Analysis:**
   - Generated n-grams (unigrams, bigrams, and trigrams) to identify common phrases and words associated with each class.
   - Visualized the most frequent n-grams for both classes to discern linguistic patterns.

6. **Word Cloud Visualization:**
   - Created word clouds for each class to visualize the most frequently occurring words, providing insights into prevalent themes and sentiments.

7. **Correlations and Relationships:**
   - Explored potential correlations between features (if applicable) to uncover any interesting relationships that could inform feature selection and engineering.

## Previous Results

Initially, we employed various preprocessing techniques and vectorization methods to classify the text data. The first set of results were:

- **Count Vectorizer:**
  - Accuracy: **86.16%**
  - This approach simply counted the occurrences of each word in the texts, which captured the basic features but may have missed the significance of word importance.

- **TF-IDF Vectorizer:**
  - Accuracy: **86.37%**
  - By considering the term frequency and inverse document frequency, this method gave more weight to less frequent but informative words, improving classification performance.

- **Word2Vec Embeddings:**
  - Accuracy: **78.63%**
  - While this method provided dense representations of words based on context, it performed slightly worse than the TF-IDF due to its reliance on more complex training processes and potential issues with dataset size.

## Improvements and New Results

After analyzing the initial results and conducting further experiments, we implemented additional optimizations:

1. **Enhanced Preprocessing:**
   - Utilized advanced cleaning techniques, including emoji replacement and expansion of contractions, improving the dataset's quality.
   - Implemented lemmatization with POS tagging to ensure that word forms were consistently represented.

2. **Model Selection and Hyperparameter Tuning:**
   - Employed `RandomizedSearchCV` for hyperparameter optimization across all classifiers, which allowed us to explore a broader range of hyperparameter values efficiently.
   - Tested multiple classifiers (e.g., Logistic Regression, Random Forest, SVM) for each vectorization method.

### New Results:

- **Count Vectorizer:**
  - New Accuracy: **91.60%** (Improved)
  
- **TF-IDF Vectorizer:**
  - New Accuracy: **92.25%** (Improved)
  
- **Word2Vec:**
  - New Accuracy: **90.64%** (Improved)

## Conclusion

By leveraging randomized search for hyperparameter optimization, we significantly enhanced the performance of our classifiers. The TF-IDF vectorizer with `Logistic Regression` emerged as the best approach, achieving an accuracy of **93.85%**. Enhanced preprocessing techniques and a focus on model optimization led to substantial improvements over the initial results. This iterative approach underscores the importance of thorough EDA and continuous model refinement in developing effective predictive models in NLP.
