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
The tutorial content in [textclassification_tutorial.ipynb](textclassification_tutorial.ipynb) is structured as follows:
1. Introduction: single-label classification task, its applications.
2. Preparation: load data, load language specific model
3. Setup: classification task, setting parameters
4. Model training preparation
   - Choose/select training parameters: batch size, learning rate, epochs etc.
   - Choose/select model tools: optimizer, learning rate scheduler, tokenizer, pretrained model instance 
5. Training
   - Implement the training loop with selected tools and parameters
   - Execute the training and validation cells
   - Store the best performing model
6. Evaluation
   - Test the model on unseen data
   - Generate classification report with metrics (precision, recall, F1-score)
   - Interpreting results, analyzing model's performance and identifying potential improvements

Ready to jump in, go to [textclassification_tutorial.ipynb](textclassification_tutorial.ipynb)

## Duration
This tutorial will take approximately 1–2 hours.

## Social Science Use Cases 
A social scientist studying social media behavior over time. A training set should be created that teaches the classification of the desired information e.g., political-leaning, factualness of argument, emotionality, stance towards a topic etc.

## How to Use
1. Clone the repository or download the notebook.
2. Set up the tutorial working environment (as explained in Environment Setup section)
3. Open the notebook [textclassification_tutorial.ipynb](textclassification_tutorial.ipynb)
4. Execute the cells sequentially to preprocess data, fine-tune the model, and perform evaluations.
5. For custom tasks: Modify hyperparameters as needed, change dataset in the notebook or link to external file and re-execute the notebook cells (step 4)

## Conclusion
This tutorial covers the end-to-end pipeline for single-label text classification using fine-tuning of a pre-trained LLM (RoBERTa) model. By following these steps, you will acquire the skills to apply use and fine-tune LLMs models for other tasks and datasets.

## References

## Additional Resources
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/)

## Contact Details
For questions or feedback, contact:

- **Stephan Linzbach**: [Stephan.Linzbach@gesis.org](mailto:Stephan.Linzbach@gesis.org)

## Acknowledgments
This tutorial uses resources from the Hugging Face library and PyTorch framework.
