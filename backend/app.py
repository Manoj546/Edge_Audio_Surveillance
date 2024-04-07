from flask import Flask, request, jsonify
import socketio

# Initialize Flask app
app = Flask(__name__)

# Initialize Socket.IO
sio = socketio.Server(cors_allowed_origins='*')
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

# Define a route to handle POST requests
@app.route('/data', methods=['POST'])
@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.json
    print("Received data:", data)  # Add this line
    result = data
    sio.emit('result', result)
    print("Emitted result:", result)  # Add this line
    return jsonify({'result': result})


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0')
