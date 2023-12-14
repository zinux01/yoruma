from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
from roboflow import Roboflow

app = Flask(__name__, template_folder='')

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/startDetection', methods=['POST'])
def start_detection():
    # Menerima gambar dari permintaan POST
    image_data = request.files['image']
    
    # Simpan gambar ke file temporary
    image_path = 'temp_image.jpg'
    image_data.save(image_path)

    # Proses deteksi menggunakan Roboflow atau metode lainnya
    rf = Roboflow(api_key="KR5n5LY4XEjN5v0eKkgD")
    project = rf.workspace().project("deteksi23")
    model = project.version(9).model
    
    result = model.predict(image_path, confidence=40, overlap=30).save("static/hasil/prediction.jpg")

    # Siapkan data respons
    response_data = {
        "image_url": "static/hasil/prediction.jpg",
    }
    # Kirim hasil deteksi (contoh: nama file prediksi) kembali ke klien
    return jsonify(response_data)
if __name__ == '__main__':
    app.run(debug=True)
