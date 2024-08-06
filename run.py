import os
from app import app

app.config['APP_PATH'] = os.path.dirname(__file__)

if __name__ == '__main__':
    app.run()