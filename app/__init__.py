import os
import secrets
from flask import Flask
from decouple import config

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

from app import routes