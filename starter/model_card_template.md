# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
We used Random Forest Classifier from Sklearn with the default parameters.
## Intended Use
This model estimates an individual's pay level by combining various features.
## Training Data
The UCI Machine Learning Repository provided the training data. The information comes from the 1994 Census database. Barry Becker gathered the information from the 1994 Census database. There are 48,842 instances and 14 attributes in the data set. It can be found at https://archive.ics.uci.edu/ml/datasets/census+income.
## Evaluation Data
The main dataset was separated into train and test with an 80/20 split for the evaulation data. The model's performance is evaluated using the test data. In terms of data processing, we employed categorical encoding with onehot encoding and binazier for the target variable.
## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations
Because the training characteristics contained information like as ethnicity and gender, model performance should be regarded and analysed with caution. This could result in skewed predictions.
## Caveats and Recommendations
This model's sole application is to forecast an individual's pay level. The model should never be utilised for anything else.