FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install .

CMD ["/usr/local/bin/start-concert-api"]
