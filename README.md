# Text Classification Tutorial

## Learning Objectives

This tutorial has the following learning objectives:
-	Learning how to work with large language models (RoBERTa)
-	Customizing (fine-tuning) a large language model for a text classification task in any language (100+ languages supported)
-	Low-resource learning (with only few hundred examples) using the SAM optimizer


## Description

This step-by-step tutorial provides an accessible introduction to customizing (fine-tuning) a pre-trained multilingual language model (RoBERTa) for text classification tasks. It demonstrates how to use the model's existing knowledge to classify text accurately, even with a small set of labeled examples. It takes input as JSON files with text documents and their corresponding labels for training, validating and testing. It covers using specialized models for English, German, and French while employing XLM-RoBERTa for over 100 additional languages.

## Target Audience (Difficulty Level)

-	Social scientists willing to learn about using large language models with basic prior understanding of it
-	Social scientists with expertise in large-language models, interested in fine-tuning for multiple languages from only few examples.
-	Computer scientists interested in learning about how large-language models are used for social text classification.
-	Advanced NLP researchers and professors looking for tutorials that can help their students in learning new topics.

## Prerequisites

Use this tutorial preferably in [Google Colab](https://colab.research.google.com/github/Stephan-Linzbach/Text-Classification-with-Pretrained-Language-Models/blob/main/textclassification_tutorial.ipynb), as the setup depends on the pre-installed packages of the Colab environment.

## Environment Setup

To follow this tutorial, ensure the following:

- Python >= 3.7 installed.
- Install required libraries by running:
  ```bash
  pip install -r requirements.txt
  ```

## Tutorial Content

### 1. Introduction to the Task
- Explanation of single-label classification and its applications.

### 2. Preparation
- Loading the data
- Loading language-specific language model

### 3. Defining the Classification Task
- Defining the way data points are classified.
- Setting the number of different labels.

### 4. Setting up the Model
- Defining training parameters: batch size, learning rate, number of epochs.
- Defining training infrastructure: optimizers, a learning rate scheduler, a model, and a tokenizer.

### 5. Training
- Implementing the training loop with learning rate scheduling and optimizer setup.
- Running the training and validation loop.
- Load the best-performing model.

### 6. Evaluation
- Testing the model on unseen data.
- Generating a classification report with metrics (precision, recall, F1-score).
- Interpreting results and identifying potential improvements.
- Understanding the classification report (precision, recall, F1-score, and accuracy).
- Analyzing the model's performance on the test set.


## Duration

This tutorial will take approximately 1–2 hours.

## Social Science Use Cases (Optional)

A social scientist studying social media behavior over time. A training set should be created that teaches the classification of the desired information: political-leaning, factualness of argument, emotionality, stance towards a topic etc.

## Flow Diagram (Optional)

The flow involves:

1. Preparation.
2. Model configuration and initialization.
3. Training and validation.
4. Evaluation and result interpretation.

## How to Use

1. Clone the repository or download the notebook.
2. Follow the steps sequentially in the notebook to preprocess data, fine-tune the model, and evaluate it.
3. Modify hyperparameters or use a different dataset for custom tasks.

## Conclusion

This tutorial covers the end-to-end pipeline for text classification using fine-tuning of a pre-trained model. By following the steps, you will acquire the skills to apply similar techniques to your own datasets and tasks.

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/)

## Contact Details

For questions or feedback, contact:

- **Stephan Linzbach**: [Stephan.Linzbach@gesis.org](mailto:Stephan.Linzbach@gesis.org)

## Acknowledgments

This tutorial uses resources from the Hugging Face library and PyTorch framework.
