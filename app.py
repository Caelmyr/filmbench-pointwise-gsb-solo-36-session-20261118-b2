import os
from pathlib import Path

from waitress import serve

from filmbench.server import register
from image_lab import create_app

data = Path(os.environ.get('FILMBENCH_DATA_DIR', 'data'))
app = create_app(data / 'preparation')
register(app, data / 'negatives')

if __name__ == '__main__':
    serve(app, host=os.environ.get('FILMBENCH_HOST', '127.0.0.1'),
          port=int(os.environ.get('PORT', '18939')), threads=3)
