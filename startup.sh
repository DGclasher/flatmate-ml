#!/bin/sh

# Train the model
python3 /app/scripts/train.py

# Start the server
gunicorn -b 0.0.0.0:5000 -w 2 app:app