from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
import uuid
import hashlib
from stegano import lsb
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# ==========================
# GLOBAL STORAGE
# ==========================
uuid_image_map = {}
latest_file_uuid = None

# ==========================
# LOAD DEEP LEARNING MODEL
# ==========================
mobilenet_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# ==========================
# DEEP LEARNING HELPER
# ==========================
def get_embedding_mask(image, threshold=0.6):
    """
    Uses MobileNetV2 to find high-texture regions
    Returns a boolean mask for embedding
    """
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    features = mobilenet_model.predict(img_array, verbose=0)
    activation_map = np.mean(features[0], axis=-1)

    # Normalize
    activation_map -= activation_map.min()
    activation_map /= (activation_map.max() + 1e-8)

    # Resize mask to original image size
    activation_map = tf.image.resize(
        activation_map[..., np.newaxis],
        image.size
    ).numpy().squeeze()

    return activation_map > threshold

# ==========================
# IMAGE HIDING (DL + LSB)
# ==========================
def hide_image_with_more_bits(cover_image, secret_image, bits=4):
    secret_image = secret_image.resize(cover_image.size)
    cover_pixels = np.array(cover_image)
    secret_pixels = np.array(secret_image)

    mask = (1 << bits) - 1
    embedding_mask = get_embedding_mask(cover_image)

    for y in range(cover_pixels.shape[0]):
        for x in range(cover_pixels.shape[1]):
            if embedding_mask[y, x]:
                for c in range(3):
                    cover_pixels[y, x, c] = (
                        (cover_pixels[y, x, c] & ~mask)
                        | (secret_pixels[y, x, c] >> (8 - bits))
                    )

    return Image.fromarray(cover_pixels)

# ==========================
# PASSWORD STEGANOGRAPHY
# ==========================
def hide_password_in_image(image, password, bits=4):
    pixels = list(image.getdata())
    password_bin = ''.join(format(ord(c), '08b') for c in password) + '00000000'
    mask = (1 << bits) - 1

    new_pixels = []
    password_idx = 0

    for pixel in pixels:
        new_pixel = []
        for channel in pixel:
            if password_idx < len(password_bin):
                new_channel = (channel & ~mask) | int(
                    password_bin[password_idx:password_idx+bits], 2
                )
                password_idx += bits
            else:
                new_channel = channel
            new_pixel.append(new_channel)
        new_pixels.append(tuple(new_pixel))

    new_image = Image.new(image.mode, image.size)
    new_image.putdata(new_pixels)
    return new_image

def extract_password_from_image(image, bits=4):
    pixels = list(image.getdata())
    password_bin = ''
    mask = (1 << bits) - 1

    for pixel in pixels:
        for channel in pixel:
            password_bin += format(channel & mask, f'0{bits}b')
            if len(password_bin) % 8 == 0 and password_bin[-8:] == '00000000':
                return ''.join(
                    chr(int(password_bin[i:i+8], 2))
                    for i in range(0, len(password_bin) - 8, 8)
                )
    return ""

# ==========================
# FLASK ROUTES (UNCHANGED)
# ==========================
@app.route('/hide_image', methods=['POST'])
def hide_image():
    cover_image_file = request.files['cover_image']
    secret_image_file = request.files['secret_image']
    password = request.form.get('password', '')

    if not password:
        return jsonify({"status": "error", "message": "Password is required"}), 400

    cover_image = Image.open(cover_image_file.stream).convert("RGB")
    secret_image = Image.open(secret_image_file.stream).convert("RGB")

    new_image = hide_image_with_more_bits(cover_image, secret_image, bits=4)
    new_image = hide_password_in_image(new_image, password, bits=4)

    img_io = io.BytesIO()
    new_image.save(img_io, 'PNG')
    img_io.seek(0)

    global latest_file_uuid
    latest_file_uuid = str(uuid.uuid4())
    uuid_image_map[latest_file_uuid] = img_io

    return jsonify({"encoded_image_url": request.url_root + 'processed'})

@app.route('/extract_image', methods=['POST'])
def extract_image():
    cover_image_file = request.files['cover_image']
    password = request.form.get('password', '')

    if not password:
        return jsonify({"status": "error", "message": "Password is required"}), 400

    cover_image = Image.open(cover_image_file.stream).convert("RGB")

    extracted_password = extract_password_from_image(cover_image, bits=4)
    if extracted_password != password:
        return jsonify({"status": "error", "message": "Invalid password"}), 401

    cover_pixels = np.array(cover_image)
    extracted_pixels = (cover_pixels & 15) * 17

    extracted_image = Image.fromarray(extracted_pixels.astype(np.uint8))

    img_io = io.BytesIO()
    extracted_image.save(img_io, 'PNG')
    img_io.seek(0)

    global latest_file_uuid
    latest_file_uuid = str(uuid.uuid4())
    uuid_image_map[latest_file_uuid] = img_io

    return jsonify({"extracted_image_url": request.url_root + 'processed'})

@app.route('/encode', methods=['POST'])
def encode_message():
    if 'file' not in request.files or 'secret_message' not in request.form or 'password' not in request.form:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    file = request.files['file']
    secret_message = request.form['secret_message']
    password = request.form['password']

    image = Image.open(file.stream)
    secret_message_with_password = f"{secret_message}||{hashlib.sha256(password.encode()).hexdigest()}"

    encoded_image = lsb.hide(image, secret_message_with_password)
    encoded_image_bytes = io.BytesIO()
    encoded_image.save(encoded_image_bytes, format='PNG')
    encoded_image_bytes.seek(0)

    global latest_file_uuid
    latest_file_uuid = str(uuid.uuid4())
    uuid_image_map[latest_file_uuid] = encoded_image_bytes

    return send_file(encoded_image_bytes, mimetype='image/png')

@app.route('/decode', methods=['POST'])
def decode_message():
    if 'file' not in request.files or 'password' not in request.form:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    image_file = request.files['file']
    password = request.form['password']

    image = Image.open(image_file.stream)
    decoded = lsb.reveal(image)

    if decoded is None:
        return jsonify({"status": "error", "message": "No hidden message found"}), 400

    secret_message, stored_hash = decoded.rsplit('||', 1)
    if hashlib.sha256(password.encode()).hexdigest() == stored_hash:
        return jsonify({"message": secret_message})
    else:
        return jsonify({"status": "error", "message": "Invalid password"}), 401

@app.route('/processed')
def send_latest_processed_file():
    if latest_file_uuid and latest_file_uuid in uuid_image_map:
        img_io = uuid_image_map[latest_file_uuid]
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    return jsonify({"error": "No processed file found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
