from flask import Flask, render_template,request,redirect,url_for,session
from flask_socketio import SocketIO,send,join_room
import random
import json

import torch

from model import NeuralNet
import nltk # UI Developer code edit
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)# UI code
app.secret_key="chatty"
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
nltk.download('punkt')# pre trained tokenizer

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('sessions.html')

@app.route('/make-reply',methods=['GET','POST'])
def make_reply():
    message=request.form['message']

    # the  chatbot script
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Sam"
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = message
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    # bot_name var
                    return {"reply":random.choice(intent['responses'])}
        else:
            return {"reply":"I do not understand..."}







if __name__ == '__main__':
    app.run(debug=True)
