
# from flask import Flask, render_template, Response, redirect, url_for, request
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# import io

# # Initialize Flask app
# app = Flask(__name__)

# # Load the YOLOv5 model
# model_path = r'C:\Users\remonda\Downloads\websit\websit\best (8).pt'  # Update this path
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# # Initialize webcam
# video_capture = cv2.VideoCapture(0)

# @app.route('/')
# def index():
#     return render_template('index.html')

# # Endpoint for real-time predictions
# @app.route('/predict')
# def predict():
#     # Function to generate frames from webcam
#     def generate_frames():
#         while True:
#             # Capture frame-by-frame
#             success, frame = video_capture.read()
#             if not success:
#                 break

#             # Convert frame to PIL Image
#             pil_img = Image.fromarray(frame)

#             # Convert PIL Image to numpy array
#             img_np = np.array(pil_img)

#             # Run the model on the image
#             results = model(img_np)

#             # Format the results
#             for *box, conf, cls in results.xyxy[0].cpu().numpy():
#                 # Draw bounding box on the frame
#                 cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#                 # Put class label and confidence score on the frame
#                 cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Encode the frame as JPEG image
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()

#             # Yield the frame in the byte format
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     # Return the generated frames as response
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/camera.html')
# def camera_html():
#     return render_template('camera.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         # Check login credentials (this is a placeholder, you should implement your actual login logic)
#         email = request.form['email']
#         password = request.form['password']
#         if email == 'example@example.com' and password == 'password':
#             # Redirect to camera page upon successful login
#             return redirect(url_for('login_success'))
#         else:
#             # Handle invalid login credentials
#             return render_template('login.html', message='Invalid credentials')
#     else:
#         return render_template('login.html')

# @app.route('/login_success')
# def login_success():
#     return render_template('camera.html')

# @app.route('/signin')
# def signin():
#     return render_template('signin.html')

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, Response, redirect, url_for, request
import torch
import cv2
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv5 model
model_path = r'C:\Users\remonda\Downloads\websit\websit\best (8).pt'  # Update this path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for real-time predictions
@app.route('/predict')
def predict():
    # Function to generate frames from webcam
    def generate_frames():
        while True:
            # Capture frame-by-frame
            success, frame = video_capture.read()
            if not success:
                break

            # Convert frame to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run the model on the image
            results = model(pil_img)

            # Format the results
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                # Draw bounding box on the frame
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # Put class label and confidence score on the frame
                cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame as JPEG image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Return the generated frames as response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera.html')
def camera_html():
    return render_template('camera.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check login credentials (this is a placeholder, you should implement your actual login logic)
        email = request.form['email']
        password = request.form['password']
        if email == 'example@example.com' and password == 'password':
            # Redirect to camera page upon successful login
            return redirect(url_for('login_success'))
        else:
            # Handle invalid login credentials
            return render_template('login.html', message='Invalid credentials')
    else:
        return render_template('login.html')

@app.route('/login_success')
def login_success():
    return render_template('camera.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)




