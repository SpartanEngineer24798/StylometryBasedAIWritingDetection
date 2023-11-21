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

    python .\feature_extraction.py -i path/to/input/csv -o path/to/output/csv

If you want to allow the program to download the dataset too, then add the --download-dataset argument at the end (may require that the program have read write access to the current directory as well as ability to access the internet)
This code may also require separate python packages of "requests" and "zipfile" be installed in your environment.

    python .\feature_extraction.py -i C:\Users\eddie\Dropbox\맏이\Projects\StylometricAI\GPT-wiki-intro.csv -o C:\Users\eddie\Dropbox\맏이\Projects\StylometricAI\extracted_features.csv --download-dataset
