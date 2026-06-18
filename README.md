# Multimodal Sentiment Analysis System 

This project is a sentiment analysis system that uses both **text and images** to determine whether a product review is:

- Positive   
- Neutral   
- Negative

## What This Project Does

The system reads a csv file with the following columns:
- Product name
- Product description
- Customer review text
- Associated review images

All of these are combined to predict the sentiment of each review.

# IMPORTANT

## Installation
1. Clone the repository
   - git clone <repository-link>
   - cd ai-project
2. Use Virtual Environment (optional but recommended)
   - Create environment:
       python -m venv .venv
   - Activate environment:
       .venv\Scripts\activate
       If using Windows (PowerShell): .venv\Scripts\Activate.ps1
3. Install the required libraries for the model to work
   - pip install -r requirements.txt
   if it doesn't work, do this instead:
   - pip install torch torchvision transformers pandas pillow pyyaml
  
## How to Start Training

Run this command:
- **python main.py**
From here, the dataset is loaded. The model starts learning and the training progress is shown per **epoch**.
The saved models appear in the following directory **outputs/checkpoints/**

## How to Evaluate the Model (with metrics)

After training, test the model:

   **python evaluate.py --csv data/reviews.csv** (or whatever csv filename you used)

If you want to use a specific model:
   
   python evaluate.py --csv data/reviews.csv --checkpoint outputs/checkpoints/model_epoch_3.pt

- replace reviews.csv with your csv filename
- replace model_epoch_3.pt with the specific model you want to use

## How to Make Predictions

To predict sentiment on a new data/dataset:

   **python predict.py --csv data/reviews.csv**

-replace reviews.csv with your csv filename 

Outputs will be saved at this directory: **outputs/predictions.csv**

## CSV File Format
Dataset must contain:
- product_title: name of product
- product_description: product details
- review_text: customer review
- image_paths: for multiple images in one review, filenames separated by semicolon(e.g. 1_1.jpg;1_2.jpg)
- label: sentiment label (0 - negative; 1 - neutral; 2 - positive)

## Notes
- Images must be stored inside **data/images/** directory
- Training must be done before evaluation or prediction
- Other files such as debug_inspect.py and test_forward.py should be ignored. These files are only kept for documentation purposes only.

## Authors
BS Computer Science Students
Polytechnic University of the Philippines

## Purpose
This project is for academic and research purposes only.
