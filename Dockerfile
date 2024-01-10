FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
WORKDIR /usr/src/app
RUN ls
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "./MNISTTrainingLoop.py"]