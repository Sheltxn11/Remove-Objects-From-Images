from flask import Flask, request, jsonify
from flask_cors import CORS
from gan_inpaint import gan_inpaint

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
            
        result = gan_inpaint(image_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)