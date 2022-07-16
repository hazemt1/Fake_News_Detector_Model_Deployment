FROM openkbs/jdk11-mvn-py3

RUN sudo apt install git

EXPOSE 8080

WORKDIR /code

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . ./

CMD ["python3", "./app.py"]