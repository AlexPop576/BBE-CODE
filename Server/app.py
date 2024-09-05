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
import os

# Define paths
image_path = "captured_image123.jpg"
owfs_path = '/mnt/1wire/'

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

from PIL import Image
import numpy as np
import pytesseract

def capture_id():
    global captured_text
    try:
        # Capture the image array from the camera
        image_array = picam2.capture_array()
        
        if image_array is not None:
            # Convert the NumPy array to a PIL Image
            image = Image.fromarray(image_array)
            image = image.convert('RGB')  # Convert to RGB mode
            
            # Get the dimensions of the image
            width, height = image.size
            
            # Define the cropping box for the left half of the image
            left_half_box = (0, 0, width // 2, height)
            
            # Crop the image to the left half
            image = image.crop(left_half_box)
            
            # Save the cropped image
            image.save(image_path, 'JPEG')
            
            # Extract text using OCR
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

@app.route('/saved-image')
def get_saved_image():
    try:
        # Check if the file exists
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return "Image not found", 404
    except Exception as e:
        print(f"Error serving image: {e}")
        return "Internal Server Error", 500


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

# Flag to indicate if the server should keep listening for iButton
listening = False
ibutton_detected = False

@app.route('/help-listen')
def start_listening():
    global listening, ibutton_detected
    listening = True
    ibutton_detected = False

    # Start a new thread to listen for the iButton
    thread = threading.Thread(target=listen_for_ibutton)
    thread.start()

    return jsonify({"status": "Started listening for iButton."})

@app.route('/ibutton-status')
def ibutton_status():
    global ibutton_detected
    return jsonify({"detected": ibutton_detected})

def listen_for_ibutton():
    global listening, ibutton_detected

    while listening:
        if check_ibutton_presence():
            ibutton_detected = True
            listening = False  # Stop listening once iButton is detected
            break

        time.sleep(1)

def check_ibutton_presence():
    owfs_path = '/mnt/1wire/'  # Define your OWFS path here
    devices = os.listdir(owfs_path)
    ibuttons = [dev for dev in devices if dev.startswith('0C.') or dev.startswith('24.') or dev.startswith('01.') or dev.startswith('2D.')]

    if ibuttons:
        print(f"iButton detected: {ibuttons}")
        return True
    else:
        print("No iButton detected.")
        return False


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
