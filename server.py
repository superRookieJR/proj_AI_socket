from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS  # Import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

# Initialize Flask and SocketIO
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow CORS for Socket.IO
model = tf.keras.models.load_model('rock_paper_scissors_model_v99.h5')

def preprocess_image(image_base64):
    # แปลง base64 string ให้เป็น bytes
    image_data = base64.b64decode(image_base64)
    
    # เปิดรูปภาพโดยใช้ PIL จากข้อมูลที่ถอดรหัสแล้ว
    img = Image.open(io.BytesIO(image_data))
    
    # Resize รูปภาพให้เป็นขนาดที่ model ต้องการ (224x224)
    img = img.resize((224, 224))
    
    # แปลงรูปภาพให้เป็น numpy array และ Normalize ค่า
    img_array = np.array(img) / 255.0
    
    # เพิ่ม batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@socketio.on('upload_image')
def handle_image(data):
    try:
        print("Received image from client...")

        # รับข้อมูล base64 image จาก client
        image_base64 = data['image']
        
        # Preprocess รูปภาพ
        img_array = preprocess_image(image_base64)

        print("Image processed, making prediction...")
        
        # ทำการทำนายคลาสของรูปภาพ
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        print(f"Predicted class: {predicted_class}")
        
        # ส่งผลลัพธ์กลับไปยัง client
        socketio.emit('image_classification', {'classification': str(predicted_class)})
    except Exception as e:
        print(f"Error: {e}")
        socketio.emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)