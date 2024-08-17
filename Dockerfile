FROM python:3.12-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn
RUN chmod +x ./startup.sh

EXPOSE 5000

CMD [ "/app/startup.sh" ]
