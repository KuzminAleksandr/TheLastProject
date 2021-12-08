FROM python:3.8-slim

WORKDIR /usr/src/app

RUN chown -R root:root /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R root:root ./

ENV SECRET_KEY hello
RUN chmod +x ./src/start_flask.py

CMD [ "python", "./src/start_flask.py" ]