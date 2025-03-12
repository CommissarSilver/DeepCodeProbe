FROM --platform=linux/amd64 python:3.9
WORKDIR /type4py/
# Put the required models files in a folder "t4py_model_files" inside "/type4py"
# -type4py/
# --type4py/
# --t4py_model_files/
COPY src /type4py
COPY ManyTypes4Py.tar.gz /type4py/

RUN tar -xvzf /type4py/ManyTypes4Py.tar.gz -C /type4py/

ENV T4PY_LOCAL_MODE="1"

# The current model files are pickled with the below ver. of sklearn
RUN pip install scikit-learn==0.24.1

RUN pip install -e .
# Web server's required packages
RUN pip install -r requirements.txt

# Install NLTK corpus
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -c "import nltk; nltk.download('omw-1.4')"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"


CMD ["tail", "-f", "/dev/null"]