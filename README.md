# StylometryBasedAIWritingDetection
Using Stylometric Features to classify if a piece of text is written by an AI or not

To use this repository, follow the steps below.

Set up your environment:

    It is recommended to use Python 3.9 version
    Create a virtual environment (venv) or conda environment with Python 3.9.
    Install all the requirements by running the following commands:

   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_trf #en_core_web_trf is a spacy module that needs to be downloaded separately
   ```

Run your code by giving the input and the output file paths and arguments to feature_extraction.py:

    python feature_extraction.py path/to/GPT-wiki-intro.csv path/to/outpot/extracted_features.csv
