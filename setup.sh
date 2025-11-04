python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 scripts/get_dataset.py
python3 scripts/verify_setup.py
pip install --upgrade pip
pip install ipykernel jupyter
python -m ipykernel install --user --name cos781 --display-name "COS781 (venv)"
echo "Setup complete. Use 'jupyter notebook' and select kernel 'COS781 (venv)'."
