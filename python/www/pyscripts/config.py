import os

# Get the directory containing the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root project directory (one level up from scripts)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define all paths relative to project root
PATH_DATA = os.path.join(PROJECT_ROOT, "pydata", "py_csv")
PATH_PLOTS = os.path.join(PROJECT_ROOT, "pydata", "pyplots")
PATH_IMAGES_TRAVEL = os.path.join(PROJECT_ROOT, "slike_travel")
PATH_IMAGES_CITIES = os.path.join(PROJECT_ROOT, "slikeGradova")
API_FILE = os.path.join(PROJECT_ROOT, "pydata", "api.txt")
# Create directories if they don't exist
os.makedirs(PATH_DATA, exist_ok=True)
os.makedirs(PATH_PLOTS, exist_ok=True)
os.makedirs(PATH_IMAGES_TRAVEL, exist_ok=True)
os.makedirs(PATH_IMAGES_CITIES, exist_ok=True)