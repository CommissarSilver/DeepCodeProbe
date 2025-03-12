FROM --platform=linux/amd64 python:3.9

WORKDIR /app

COPY src /app/

RUN pip install -r DEAR/approach/requirements.txt

CMD ["tail", "-f", "/dev/null"]
