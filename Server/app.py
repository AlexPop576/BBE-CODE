from flask import Flask, render_template, send_file, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import usb.core
import usb.util
from PIL import Image
import cv2
import numpy as np
import logging
from flask_cors import CORS
import pytesseract
from io import BytesIO
from picamera2 import Picamera2

# Define paths
image_path = "captured_image.jpg"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Global variables
current_frame = None
captured_text = None
cameraStarted = False
frame_lock = threading.Lock()
capture_interval = 0.2  # Capture interval in seconds
capture_thread = None
stop_event = threading.Event()
device = None
usb_lock = threading.Lock()

# Define your USB device's vendor ID and product ID
VENDOR_ID = 0x04fa  # Replace with your device's vendor ID
PRODUCT_ID = 0x2490  # Replace with your device's product ID

# Initialize OpenCV and Webcam for video streaming
webcam = None

# Initialize Picamera2 for text capture
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XBGR8888", "size": (2048, 1080)})
picam2.configure(config)

# Global dictionary to track votes
votes = {}
# Set to track CNPs of users who have voted
already_voted_cnp = set()

def start_webcam():
    global webcam
    if webcam is None or not webcam.isOpened():
        # Check for the first available webcam
        for index in range(5):  # Try up to 5 different indices
            test_cam = cv2.VideoCapture(index)
            if test_cam.isOpened():
                webcam = test_cam
                print(f"Webcam started on index {index}")
                break
            test_cam.release()
        if webcam is None or not webcam.isOpened():
            print("Error: No webcam found or camera index out of range.")

def stop_webcam():
    global webcam
    if webcam is not None and webcam.isOpened():
        webcam.release()
        webcam = None
        print("Webcam released.")

def capture_image():
    global current_frame
    while not stop_event.is_set():
        try:
            if webcam is None or not webcam.isOpened():
                start_webcam()
            if webcam is not None and webcam.isOpened():
                ret, image_array = webcam.read()  # Capture frame-by-frame from the webcam
                if ret:
                    with frame_lock:
                        current_frame = image_array
                    socketio.emit('image_update')
            time.sleep(capture_interval)
        except Exception as e:
            print(f"Error capturing image: {e}")

def capture_id():
    global captured_text
    try:
        image_array = picam2.capture_array()
        if image_array is not None:
            image = Image.fromarray(image_array)
            captured_text = pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error in capture_id: {e}")
        captured_text = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def get_image():
    global current_frame
    with frame_lock:
        if current_frame is not None:
            img = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
            img_io = BytesIO()
            img.save(img_io, 'JPEG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
        else:
            return "No image captured", 404

@app.route('/capture')
def capture_and_get_text():
    global captured_text
    capture_id()
    print(f"Captured Text: {captured_text}")
    return jsonify({'captured_text': captured_text + "+"})

@app.route('/start-camera')
def start_camera():
    global cameraStarted
    global stop_event
    if not cameraStarted:
        stop_event.clear()
        start_webcam()  # Ensure the webcam is started
        if webcam is not None and webcam.isOpened():
            cameraStarted = True
            global capture_thread
            capture_thread = threading.Thread(target=capture_image, daemon=True)
            capture_thread.start()
            print("Camera started")
    return "Camera started"

@app.route('/stop-camera')
def stop_camera():
    global cameraStarted
    global stop_event
    if cameraStarted:
        stop_event.set()
        if capture_thread is not None:
            capture_thread.join()  # Wait for the thread to finish
        stop_webcam()  # Ensure the webcam is stopped
        cameraStarted = False
        print("Camera stopped")
    return "Camera stopped"

@app.route('/receive-data', methods=['POST'])
def receive_data():
    global votes
    global already_voted_cnp
    data = request.json  # Parse the incoming JSON data
    vote_name = data.get('voteName')
    vote_party = data.get('voteParty')
    cnp = data.get('cnp')

    # Add the CNP to the already voted set
    already_voted_cnp.add(cnp)

    # Check if the vote_name already exists in the votes dictionary
    if vote_name in votes:
        # If it exists, update the details if needed and increment the count
        votes[vote_name]['count'] += 1
    else:
        # Otherwise, create a new entry
        votes[vote_name] = {
            'count': 1,
            'voteParty': vote_party
        }

    print(f"Received data: {data}")
    
    # Send a response back to the client
    return jsonify({"status": "success", "received": data})

@app.route('/get-votes')
def get_votes():
    # Transform the votes dictionary to a list of dictionaries for better frontend handling
    votes_list = [{"voteName": name, "count": details['count'], "voteParty": details['voteParty']} for name, details in votes.items()]
    return jsonify(votes_list)

@app.route('/already-voted')
def already_voted():
    # Transform the already_voted_cnp set to a list of dictionaries
    already_voted_list = [{"cnp": cnp} for cnp in already_voted_cnp]
    return jsonify(already_voted_list)

@app.route('/receive-ibutton-data', methods=['POST'])
def receive_ibutton_data():
    data = request.json  # Parse the incoming JSON data
    ibutton_id = data.get('ibutton_id')
    if ibutton_id:
        print(f"Received iButton data: {ibutton_id}")
        # Process iButton data here
        return jsonify({"status": "success", "received": ibutton_id})
    else:
        return jsonify({"status": "error", "message": "No iButton ID received"}), 400

@app.route('/help-listen')
def start_listening():
    return "Started listening"

def find_device():
    global device
    with usb_lock:
        device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if device is None:
            raise ValueError('Device not found')

def setup_device():
    global device
    if device is None:
        find_device()
    device.set_configuration()
    usb.util.claim_interface(device, 0)

@app.route('/connect-usb', methods=['GET'])
def connect_usb():
    try:
        setup_device()
        return jsonify({"status": "Device connected"})
    except Exception as e:
        logging.error(f"Error connecting to USB device: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/read-usb', methods=['GET'])
def read_usb():
    global device
    if device is None:
        return jsonify({"status": "error", "message": "Device not connected"})
    try:
        # Replace with appropriate endpoint and length
        endpoint = 0x81  # Replace with your endpoint address
        length = 64
        data = device.read(endpoint, length)
        return jsonify({"status": "success", "data": data.hex()})
    except Exception as e:
        logging.error(f"Error reading from USB device: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/disconnect-usb', methods=['GET'])
def disconnect_usb():
    global device
    if device is None:
        return jsonify({"status": "error", "message": "Device not connected"})
    try:
        usb.util.release_interface(device, 0)
        usb.util.dispose_resources(device)
        device = None
        return jsonify({"status": "Device disconnected"})
    except Exception as e:
        logging.error(f"Error disconnecting USB device: {e}")
        return jsonify({"status": "error", "message": str(e)})

@socketio.on('connect')
def handle_connect():
    emit('image_update')

if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    picam2.start()  # Start the PiCamera2 for text capture
    socketio.run(app, host='0.0.0.0', port=5000)

    # Release the webcam when the app is closed
    stop_webcam()
