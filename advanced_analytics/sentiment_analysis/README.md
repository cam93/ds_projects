# Customer Sentiment Analysis Using RNN and NLP

This project uses Recurrent Neural Networks (RNN) and Natural Language Processing (NLP) techniques to perform sentiment analysis on customer reviews. By categorizing feedback as positive or negative, this model enables businesses to gain actionable insights into customer satisfaction, helping drive improvements in products and services based on real-time sentiment analysis.

## Project Overview

### Research Question
**"Can neural networks and NLP techniques effectively analyze customer sentiment to enhance customer satisfaction?"**

### Objectives
1. **Develop a Sentiment Classification Model**: Train a model on labeled customer reviews to predict sentiment (positive/negative) in future feedback.
2. **Enable Proactive Customer Service**: Use sentiment predictions to address customer concerns and recognize positive trends.
3. **Support Data-Driven Decisions**: Provide insights to guide improvements in customer experience and satisfaction.

## Methodology

### Data Preprocessing
- **Text Cleaning**: Removed unusual characters, punctuation, and stopwords; expanded contractions and converted text to lowercase.
- **Tokenization and Lemmatization**: Tokenized text into individual words, applied lemmatization to group word variants.
- **Padding and Sequencing**: Applied post-padding to standardize sequence lengths to 18 words for input into the neural network.

### Model Architecture
- **Layers**:
  - **Embedding Layer**: Transforms words into dense vectors of 32 dimensions.
  - **Bidirectional LSTM Layer**: Captures context from both directions of text sequences.
  - **Dropout Layer**: Reduces overfitting with a dropout rate of 60%.
  - **Dense Layers**: Includes a hidden layer with ReLU activation and a final output layer with sigmoid activation for binary classification.
- **Hyperparameters**:
  - Activation functions: ReLU (hidden layer) and sigmoid (output layer)
  - Optimizer: Adam with a learning rate of 0.001
  - Loss Function: Binary cross-entropy for classification accuracy

### Model Evaluation
- **Metrics**: Accuracy, precision, recall, F1-score
- **Final Performance**:
  - **Test Accuracy**: 82.23%
  - **Balanced Precision and Recall**: Ensures effectiveness in distinguishing between positive and negative sentiment.

## Key Findings

1. **Accurate Sentiment Classification**: The model effectively identifies positive and negative feedback with high accuracy.
2. **Model Robustness**: Techniques like dropout and L2 regularization helped prevent overfitting and enhance model generalization.

## Recommendations

1. **Deploy for Real-Time Feedback**: Integrate the model for continuous monitoring of customer sentiment across feedback channels.
2. **Expand to Multi-Language Support**: Adapt the model for global operations by incorporating multiple language capabilities.
3. **Use Insights for Proactive Actions**: Act on negative feedback promptly and leverage positive trends for targeted marketing.
4. **Regular Model Updates**: Retrain the model periodically to capture evolving language patterns and maintain accuracy.

## Files and Dependencies

- **data_cleaning.ipynb**: Jupyter notebook with the complete data preprocessing, model building, and evaluation code.