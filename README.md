# IMDB Multiclass Sentiment Classification

This project extends the traditional binary sentiment classification of the IMDB movie reviews dataset to a more nuanced five-class framework. Instead of simply classifying reviews as positive or negative, this model categorizes them into five distinct sentiment classes: **Very Negative**, **Negative**, **Neutral**, **Positive**, and **Very Positive**. This approach provides more detailed and informative predictions, contributing to a deeper understanding of textual sentiment.

The project follows a systematic deep learning workflow, including data preprocessing, advanced techniques for handling class imbalance, and building a hybrid neural network architecture for classification.

## Key Features

-   **Multiclass Framework:** Transforms the binary IMDB dataset into a five-class sentiment problem using a custom polarity-based heuristic re-labeling technique.
-   **Hybrid CNN-LSTM Model:** Utilizes a hybrid architecture combining a Convolutional Neural Network (CNN) for feature extraction and a Bidirectional Long Short-Term Memory (LSTM) network to capture sequential context.
-   **Advanced Imbalance Handling:** Implements a multi-stage strategy to combat class imbalance, including Random Oversampling, synonym-based text augmentation, and class weighting during training.
-   **Intelligent Training:** Employs Keras callbacks like `EarlyStopping` and `ReduceLROnPlateau` for efficient training and to prevent overfitting.
-   **Detailed Evaluation:** Provides a comprehensive performance analysis using a classification report, confusion matrices, and an in-depth look at misclassified examples.

## Methodology

The project workflow is broken down into the following key stages:

### 1. Data Preprocessing and Re-labeling

-   The standard Keras IMDB dataset is loaded, containing pre-encoded integer sequences of movie reviews.
-   A custom **heuristic re-labeling** function (`polarity_based_labels`) categorizes each review into one of the five sentiment classes based on the frequency of positive and negative words identified from NLTK's `opinion_lexicon`.
-   All sequences are padded to a uniform length of 150 using `pad_sequences` to ensure consistent input size for the model.

### 2. Class Imbalance Handling

A multi-step strategy was used to address the skewed class distribution that resulted from the re-labeling process:

1.  **Random Oversampling:** The `imblearn` library's `RandomOverSampler` is used to increase the sample count of the minority "Very Negative" and "Very Positive" classes.
2.  **Synonym Augmentation:** A custom function (`synonym_replacement`) leverages NLTK's `wordnet` to create new, augmented text samples for the extreme classes by replacing non-stopwords with their synonyms.
3.  **Class Weighting:** During model training, `compute_class_weight` from Scikit-learn is used to assign higher weights to under-represented classes, forcing the model to pay more attention to them.

### 3. Model Architecture

A hybrid neural network was constructed using the Keras Sequential API with the following layers:

-   `Embedding`: Converts word indices into dense vectors of a fixed size (`embedding_dim = 100`).
-   `Conv1D`: A 1D convolutional layer to extract local features from the sequence of word embeddings.
-   `BatchNormalization`: To stabilize and accelerate the training process.
-   `Bidirectional(LSTM)`: Captures contextual information from both forward and backward directions in the sequence.
-   `GlobalMaxPooling1D`: Reduces the feature maps to a single vector.
-   `Dense` layers with `relu` activation, `l2` regularization, and `Dropout` to prevent overfitting.
-   A final `Dense` layer with `softmax` activation for multiclass probability output.

### 4. Training and Evaluation

-   The model is compiled with the `Adam` optimizer and `categorical_crossentropy` loss function.
-   Training is made more efficient using callbacks:
    -   `EarlyStopping`: Halts training when validation loss stops improving to prevent overfitting.
    -   `ReduceLROnPlateau`: Automatically reduces the learning rate when validation loss stagnates.
-   The final model is evaluated on the test set using a **classification report** (precision, recall, F1-score) and visualized with a **confusion matrix** and **misclassification frequency matrix**.

## Results

-   The final model achieved an overall accuracy of **67.9%** on the test set.
-   The classification report and confusion matrix show strong performance for the "Very Neg" and "Very Pos" classes but also highlight challenges in distinguishing between adjacent classes (e.g., "Neutral" vs. "Positive").
-   The primary limitation identified was the inherent ambiguity and noise introduced by the polarity-based heuristic labeling method.

## Technologies Used

-   **Python 3**
-   **TensorFlow 2.18.0**
-   **Keras**
-   **Scikit-learn**
-   **Pandas**
-   **NumPy 1.26.4**
-   **Imbalanced-learn (imblearn)**
-   **NLTK**
-   **Matplotlib & Seaborn**
-   **Jupyter Notebook**

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    Run the following in a Python interpreter to download the necessary NLTK packages:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('opinion_lexicon')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    ```

## How to Run

The entire analysis and model training process is contained within the Jupyter Notebook.

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open and Run the Notebook:**
    Open the `IMDB_Classification.ipynb` file and run the cells sequentially to execute the project workflow.

3.  **Saved Model:**
    The trained model is saved as `imdb_multiclass_cnn_lstm_model.keras` upon completion of the notebook.


## References

[1] Chollet, F. (2021). *Deep Learning with Python*, 2nd ed., Manning Publications.

[2] Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, 2nd ed., O'Reilly Media.
