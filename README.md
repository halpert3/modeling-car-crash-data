# Overview

Made for a Flatiron School data science course, the idea behind this project was to practice machine learning models. 

The Jupyter notebooks for this repository are in four groups:

1. Dataframe manipulation and exploratory data analysis
2. Running various models with their default parameters 
3. Refining chosen models
4. Creating visuals for a presentation

Link to [Medium blog post about this project.](https://halpert3.medium.com/my-introduction-to-machine-learning-models-afad8595598d)

# Data

This project uses City of Chicago accident data, mostly from 2017 to 2020. I merged the main dataset with another Chicago dataset of people involved in the car crashes.

- The complete dataset of 461,315 entries was pared to 70,000 for manageability's sake and then ultimately to 55,766 after removing outliers, missing entries, etc. 
- Final columns included speed limit, weather, lighting, roadway conditions, number of vehicles involved, accident time (hour, day of week, and month), and age and sex of driver
- The modeling target "serious accident" is a composite of crashes with either “fatal” and “incapacitating” injuries.

# Process

## Data Transformation

Columns with categorical data were transformed with pandas "get_dummies" function for easier quantification and comparison.

The dataset was imbalanced; only 1.85% of accidents were "serious accidents." After splitting the data into training and testing sets, I used SMOTE to re-balance the training set, which I then used for modeling.

## Initial Modeling

I ran 11 models with their default parameters:

- Dummy Classifier (as a baseline)
- Logistic Regression
- KNN
- Naive Bayes
- Decision Trees
- Bagged Trees
- Random Forest
- SVM
- AdaBoost Trees
- Gradient Boosting
- XGBoost

## Model Assessment

When assessing the models, I chose to prioritize the metric of <u>recall</u> instead of accuracy in order to prioritize serious accidents, even if some non-serious accidents ended being misclassified as serious (the idea being to maximize public safety). Also when comparing training and testing metrics, I looked to avoid models that overfit the training data.

## Model Refinement

After the initial modeling process, I chose two models to refine, Naive Bayes and Random Forest. I ran around eight or nine variations of each, often with scikit-learn's GridSearchCV module to facilitate experimentation with various parameters. 

# Conclusion and Next Steps

Although I intentionally prioritized recall with the awareness that I'd end up some false positives, the test data of the refined models ended up with too many false positives for the models to be truly useful. In addition, discrepancies of the metrics between the training data and testing data showed even the refined models still had an overfitting problem. 

Given more time, my next steps would be to:

- run the models with different parameters to optimize for other metrics besides recall (such as F1)
- experiment with various parameters to minimize overfitting

With a more powerful computer, I would also try running the models with the entire dataset. (For this project, I ultimately used only  around 15% of it.) 
