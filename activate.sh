# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Echo message
echo "Virtual environment activated. You can now run the script with 'python3 main.py'"