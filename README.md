# Can we detect if a patient has diabetes based off their medical data?

## Project Description

This repository was created to practise the learnings from D100 Fundamentals of Data Science from Cambridge University, and also write my medium article to consolidate the content. Using data from Kaggle, I wanted to understand from the small number of numerical medical markers, if it was enough to predict if a patient has diabetes or not.

## What challenges did you face?

I faced a few issues with the domain knowledge. The Kaggle resource is limited with only a few thousand rows of data. The data itself was biased towards the non-diabetes patients. I did not run into many technical issues as I learnt my mistakes from my coursework submission. I also used a few templates from that submission in this piece of work so it was a little easier.

One key mistake I made was after the entire pipeline was done, and I came onto evaluation, I realised the AUC for LGBM Best (CV trained) was 1 which is absolutely perfect... but this is a clear sign of data leakage. I was trying to understand where this was, then I realised I cleaned the data before spliting which is a rookie mistake! Always split the train and test before cleaning the data.
TODO: Sajid to look at examples on Youtube walkthroughs regarding the train test split then cleaning the data and preparing it for model training and validation.

## How to install the project
### For Development

To create your own virtual environment:

```python
python3.12 -m venv .venv
source .venv/bin/activate

```

To create a new branch on GitHub:

```python
git checkout -b data
```

Run pre-commit packages to have clean, best practise code:

```python
pre-commit install
```

To start on development:
```python
pip install -e .
```


### For production

To run the full workflow:
```python
python -m scripts
```

You can find the individual pieces of the workflow under scripts:
1. model_training.py
2. evaluation.py
3. visualisation.py

## Data Source

The data was taken from this Kaggle Repository: https://www.kaggle.com/datasets/mathchi/diabetes-data-set. The dataset is called via API. You will need to set up your own `.env` file that contains an API key from Kaggle, which you can follow instructions for here: https://www.kaggle.com/discussions/getting-started/524433.

## Evaluation

### Evaluation Metrics

| Metric | GLM Baseline (Test) | GLM Baseline (Train) | LGBM Baseline (Test) | LGBM Baseline (Train) | GLM Best (Test) | GLM Best (Train) | LGBM Best (Test) | LGBM Best (Train) | Shuffled LGBM (Test) | Shuffled LGBM (Train) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Preds** | 0.3434 | 0.3455 | 0.3347 | 0.3456 | 0.3431 | 0.3455 | 0.3311 | 0.3455 | 0.3424 | 0.3456 |
| **Mean Outcome** | 0.3375 | 0.3455 | 0.3375 | 0.3455 | 0.3375 | 0.3455 | 0.3375 | 0.3455 | 0.3375 | 0.3455 |
| **Abs Bias** | 0.0058 | 0.0000 | -0.0028 | 0.0000 | 0.0056 | 0.0000 | -0.0064 | -0.0000 | 0.0048 | 0.0001 |
| **Bias** | 0.0173 | 0.0000 | -0.0084 | 0.0001 | 0.0166 | 0.0000 | -0.0191 | -0.0000 | 0.0143 | 0.0002 |
| **MSE** | 0.1514 | 0.1497 | 0.0188 | 0.0018 | 0.1513 | 0.1498 | 0.0174 | 0.0002 | 0.2770 | 0.2760 |
| **RMSE** | 0.3891 | 0.3870 | 0.1370 | 0.0420 | 0.3889 | 0.3870 | 0.1319 | 0.0139 | 0.5263 | 0.5254 |
| **MAE** | 0.2983 | 0.2987 | 0.0521 | 0.0255 | 0.2995 | 0.3000 | 0.0296 | 0.0086 | 0.4550 | 0.4580 |
| **Log Loss** | 0.4606 | 0.4599 | 0.0874 | 0.0265 | 0.4605 | 0.4600 | 0.1047 | 0.0087 | 0.7977 | 0.7917 |
| **Brier** | 0.1514 | 0.1497 | 0.0188 | 0.0018 | 0.1513 | 0.1498 | 0.0174 | 0.0002 | 0.2770 | 0.2760 |
| **AUC** | 0.8416 | 0.8508 | 0.9904 | 1.0000 | 0.8423 | 0.8507 | 0.9769 | 1.0000 | 0.4679 | 0.4790 |
| **Gini** | 0.6831 | 0.7016 | 0.9809 | 1.0000 | 0.6847 | 0.7014 | 0.9538 | 1.0000 | -0.0642 | -0.0419 |

### Evaluation Summary

I went through a lot of changes to understand why the AUC for LGBM is 0.99, a unrealistic success. GLM achieved a 0.84 AUC score which is more realistic and would cope better with unseen data.

I took a few steps to validate the 0.99 result:
1. Shuffled target test so the features are not blatant answers to the unseen data. The shuffled model dropped to 0.50 AUC.
2. Feature importance shows that not one feature dominates the others, Glucose had about 0.48 correlation to the target so it's more predictive rather than definite.
3. I moved the cleaning and adding features logic inside the training flow to ensure the test set remained unseen.

Overall, I believe GLM is more robust and interpretable compared to LGBM.

## Next steps

- If I was to continue this project, I would like to see more data to have a stronger prediction.
- Due to the coursework requirements, I was constricted to GLM and LGBM models but I would test out Random Forest and XGBoost.

## Licenses, Authors and Acknowledgement

### License
MIT License

Copyright (c) 2025 Sajid Ahmed

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
