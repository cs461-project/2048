# Create venv if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Echo message
echo "Virtual environment created and requirements installed. You can now run the script with 'python3 main.py'"
