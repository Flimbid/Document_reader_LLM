from flask import Flask, jsonify , request
from chatllm import chat_response

app = Flask(__name__)

@app.route('/chat', methods=['POST'])

def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    response = chat_response(user_message)
    return jsonify({"response": response})
   


if __name__ == '__main__':
    app.run(debug=True)