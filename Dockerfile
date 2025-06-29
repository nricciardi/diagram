FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN curl -fsSL https://d2lang.com/install.sh | sh -s --

RUN pip3 install torch opencv-python matplotlib requests pillow pandas torchvision numpy shapely transformers sentencepiece protobuf torchmetrics scikit-learn

CMD ["python", "main.py"]



