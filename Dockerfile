FROM python:3.6.5

 

COPY /API /app

RUN echo $(ls -1 /app)

RUN pip install --upgrade pip

RUN pip install -r /app/requirements.txt
RUN pip install tensorflow
 

# set the working directory to the directory with the app

WORKDIR /app

EXPOSE 5000

CMD python server.py
