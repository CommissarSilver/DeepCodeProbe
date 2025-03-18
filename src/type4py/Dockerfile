FROM --platform=linux/amd64 python:3.9
WORKDIR /type4py/

# Create necessary directories
RUN mkdir -p /type4py/dataset
RUN mkdir -p /type4py/t4py_model_files

# Copy the tar file
COPY ManyTypes4Py.tar.gz /type4py/
COPY src/setup.py /type4py/
COPY src/requirements.txt /type4py/

# Extract the tar file
RUN tar -xvzf /type4py/ManyTypes4Py.tar.gz -C /type4py/dataset

ENV T4PY_LOCAL_MODE="1"

# # The current model files are pickled with the below ver. of sklearn
RUN pip install scikit-learn==0.24.1
# Web server's required packages
RUN pip install -r requirements.txt

# Install NLTK corpus
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -c "import nltk; nltk.download('omw-1.4')"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

CMD ["tail", "-f", "/dev/null"]