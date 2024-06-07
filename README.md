# Stop Words Counting and Score Calculation

This repository contains a Python project designed to analyze text similarity using various metrics such as TF (Term Frequency) and Jaccard Similarity. It provides tools to calculate the similarity scores between pairs of sentences and relies on stop word removal for text normalization.

## Project Structure

- `/data`: Directory containing the dataset files, including `SimilarityCorpusSamples.xlsx`.
- `/Output`: Directory where the outputs like normalized data files will be stored.
- `/stop_words`: Critical directory containing stop word lists. **Do not replace or remove this folder** unless you are fully aware of the consequences.
- `index.py`: Main Python script for executing the similarity analysis.
- `requirements.txt`: List of Python package dependencies for the project.


## Warning

**% DO NOT Replace OR Remove The 'stop_words' Folder %**

This folder is essential for the correct operation of the script. It contains the list of stop words used during the text normalization process.


## Setup

To run this project, follow these steps:

1. Clone the repository to your local machine.
2. Ensure that Python 3.6+ is installed on your system.
3. Install the required dependencies listed in `requirements.txt` using the following command:
   
   ```bash
   pip install -r requirements.txt

## Usage

Execute the main script with the following command:

    python index.py

The script will perform the similarity analysis and save the results in the /Output directory. A vocabulary dictionary will also be saved to Dic.txt.
## Warning

The stop_words directory is an integral part of the project. Removal or modification of this directory can lead to incorrect analysis results.

## Mistakes and Corrections

To err is human, and nobody likes a perfect person! If you come across any mistakes or if you have questions, feel free to raise an issue or submit a pull request. Your contributions to improving the content are highly appreciated. Please refer to GitHub contributing guidelines for more information on how to participate in the development.

## Contact

    GitHub: cloner174
    Email: cloner174.org@gmail.com
