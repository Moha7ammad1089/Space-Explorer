# utils.py
# ملف الأدوات المساعدة

import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

def load_tflite_engine():
    """تحميل ملف الموديل .tflite تلقائياً"""
    try:
        files = [f for f in os.listdir('.') if f.endswith('.tflite')]
        if files:
            interpreter = tf.lite.Interpreter(model_path=files[0])
            interpreter.allocate_tensors()
            return interpreter
        return None
    except: return None

def tflite_predict(interpreter, image):
    """تنفيذ التوقع على الصورة"""
    target_size = (224, 224)
    img_padded = ImageOps.pad(image, target_size, color='black')
    input_data = np.array(img_padded, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def check_brightness(image):
    """حساب سطوع الصورة للتمييز بين الشمس والفضاء المظلم"""
    gray_img = image.convert('L')
    return np.array(gray_img).mean()