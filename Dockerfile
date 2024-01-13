FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "./SimpleGenerator.py"]