FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY core core

COPY src src

RUN curl -fsSL https://d2lang.com/install.sh | sh -s --

RUN pip3 install torch opencv-python matplotlib requests pillow pandas torchvision numpy shapely transformers sentencepiece protobuf torchmetrics scikit-learn

ENTRYPOINT ["python", "src/main.py"]



