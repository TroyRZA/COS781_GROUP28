python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 scripts/get_dataset.py
python3 scripts/verify_setup.py