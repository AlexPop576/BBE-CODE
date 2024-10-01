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
import face_recognition
import easyocr
from matplotlib import pyplot as plt 
import random


# Define paths
image_path = "cropped_image.jpg"
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
        # Try multiple indices to find the available webcam
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

#==================================================================
# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding to improve text detection
    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)

    return image, thresholded_image

# Function to display image with detected text
def display_image_with_boxes(image, result):
    for (bbox, text, prob) in result:
        # Get the bounding box coordinates
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # Draw bounding boxes around detected text
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Put the detected text above the bounding box
        cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the image with bounding boxes and text
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main function to perform OCR
def ocr_on_image(image_path):
    # Initialize EasyOCR reader with English language
    reader = easyocr.Reader(['en'])

    # Preprocess the image (grayscale, thresholding)
    original_image, preprocessed_image = preprocess_image(image_path)

    # Perform OCR on the preprocessed image
    result = reader.readtext(preprocessed_image)

    # Print the OCR results
    for (bbox, text, prob) in result:
        print(f"Text: '{text}', Confidence: {prob:.2f}")

    # Display the original image with bounding boxes for the detected text
    display_image_with_boxes(original_image, result)
#==================================================================

from PIL import Image
import numpy as np
import pytesseract

def extract_digits(text):
    # Use a comprehension to extract only digits
    if "M" or "m" in text:
        start = "1"
    elif "F" or "f" in text:
        start = "2"
    else:
        start = "0"
    c=0
    digits = start+"".join(char for char in text if char.isdigit())
    if len(digits) >= 14:
        return f"{digits[-13:-7]}"+"125"+f"{digits[-4:-1]}"
    else: return digits

def capture_id():
    global captured_text
    try:
        # Capture the image array from the camera
        image_array = picam2.capture_array()

        if image_array is not None:
            # Convert the NumPy array to a PIL Image
            image = Image.fromarray(image_array)

            # Rotate the image by 180 degrees
            image = image.rotate(180)

            # Create a copy for processing
            cnp_image = image.copy()

            # Convert to RGB mode for OpenCV
            cnp_image = cnp_image.convert('RGB')
            cnp_image = np.array(cnp_image)  # Convert to NumPy array for OpenCV processing

            # Ensure the array is in the right format for OpenCV (BGR)
            cnp_image = cv2.cvtColor(cnp_image, cv2.COLOR_RGB2BGR)

            # Define the cropping box for the right half of the image
            width, height = cnp_image.shape[1], cnp_image.shape[0]
            right_half_x_start = 200
            right_half_x_end = width - 50
            top_cut = 420  # Number of pixels to cut from the top

            # Ensure the cropping box is within the image dimensions
            cnp_crop = (right_half_x_start, top_cut, right_half_x_end, height - 100)
            cnp_image = cnp_image[top_cut:height - 100, right_half_x_start:right_half_x_end]

            # Save the rotated cnp_image for verification
            cv2.imwrite('image11.jpg', cnp_image)

            # Convert to grayscale
            gray = cv2.cvtColor(cnp_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian Blur to remove noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Thresholding to create a binary image
            print(cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Optionally apply dilation or erosion
            kernel = np.ones((3, 3), np.uint8)  # Increased kernel size for better results
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            thresh = cv2.erode(thresh, kernel, iterations=1)

            # Convert back to PIL format and run OCR
            final_image = Image.fromarray(thresh)
            captured_text = extract_digits(pytesseract.image_to_string(final_image, lang='eng'))
            final_image.save('fim.jpg', 'JPEG')

            print("Captured text:", captured_text)

            # Define the cropping box for the right half of the right half
            right_half_x_start = 300
            right_half_x_end = width // 3 + 120
            top_cut = 0  # Number of pixels to cut from the top

            # Ensure y_end is within the image height
            right_half_box = (right_half_x_start, top_cut, right_half_x_end, 450)

            # Crop the image to the defined box
            cropped_image = image.crop(right_half_box)

            # Convert the cropped image to RGB before saving
            cropped_image = cropped_image.convert('RGB')

            # Save the cropped image
            cropped_image.save('cropped_image.jpg', 'JPEG')

    except Exception as e:
        print(f"Error in capture_id: {e}")
        captured_text = None

####################################################

def find_face_encodings(image_path):
    # reading image
    image = cv2.imread(image_path)
    # get face encodings from the image
    face_enc = face_recognition.face_encodings(image)
    # return face encodings
    return face_enc[0] if face_enc else None

def capture_image_from_camera(save_path):
    global webcam

    # Ensure the webcam is started
    if webcam is None or not webcam.isOpened():
        start_webcam()

    if webcam is None or not webcam.isOpened():
        print("Error: Could not open camera.")
        return None  # Return None if the camera couldn't be opened

    # Capture a single frame
    ret, frame = webcam.read()

    if ret:
        # Save the captured frame to a file
        cv2.imwrite(save_path, frame)
        print(f"Image saved to {save_path}")
        return save_path  # Return the save path on success
    else:
        print("Error: Could not capture image.")
        return None

@app.route('/valid')
def face_validation():
    # Capture an image from the camera
    captured_image_path = capture_image_from_camera("captured_image_recognition.png")
    
    if captured_image_path is None:
        return jsonify({
            "status": "error",
            "message": "Could not capture image from camera."
        }), 500

    # Getting face encodings for the first image
    image_1 = find_face_encodings("cropped_image.jpg")
    
    # Getting face encodings for the captured image
    image_2 = find_face_encodings(captured_image_path)

    if image_1 is not None and image_2 is not None:
        # Checking if both images are the same
        is_same = face_recognition.compare_faces([image_1], image_2)[0]
        print(f"Is Same: {is_same}")

        if is_same:
            # Finding the distance level between images
            distance = face_recognition.face_distance([image_1], image_2)
            distance = round(distance[0] * 100)

            # Calculating accuracy level between images
            accuracy = 100 - round(distance)
            print(f"The images are the same. Accuracy Level: {accuracy}%")
            
            # Return a valid JSON response with accuracy level
            return jsonify({
                "status": "success",
                "message": "The images are the same.",
                "accuracy": accuracy
            }), 200
        else:
            print("The images are not the same.")
            
            # Return a valid JSON response indicating mismatch
            return jsonify({
                "status": "failure",
                "message": "The images are not the same."
            }), 200
    else:
        # Error case when face encodings cannot be obtained
        print("Error: Could not get face encodings for one or both images.")
        
        # Return an error response
        return jsonify({
            "status": "error",
            "message": "Could not get face encodings for one or both images."
        }), 500

##################################################

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
    cnp = captured_text
    return jsonify({'captured_text': cnp})

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
        if os.path.exists("cropped_image.jpg"):
            return send_file("cropped_image.jpg", mimetype='image/jpeg')
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
    # Transform the votes dictionary to a list of dictionaries
    votes_list = [{"voteName": name, "count": details['count'], "voteParty": details['voteParty']} for name, details in votes.items()]
    
    # Sort the votes by count in descending order
    votes_list.sort(key=lambda x: x['count'], reverse=True)
    
    return jsonify(votes_list)

@app.route('/already-voted')
def already_voted():
    # Transform the already_voted_cnp set to a list of dictionaries
    already_voted_list = [{"cnp": cnp} for cnp in already_voted_cnp]

    # Randomize the order of the CNPs
    random.shuffle(already_voted_list)
    
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
