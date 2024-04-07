API call (Postman) device to server:

curl --location 'http://192.168.18.147:5000/data' \
--header 'Content-Type: application/json' \
--data '{"text": "nice", "audio": "nice"}'


Running frontend:

cd ./frontend
npm i
npm start

Running Backend:

cd ./backend
pip install virtualenv venv
./venv/Scripts/activate.ps1
pip install -r requirements.txt
pip install flask socketio
python ./app.py