
# Importing Necessary Modules
import random
from flask import Flask, render_template,request,redirect,url_for,session
from flask_socketio import SocketIO,send,join_room
import torch
import json
import nltk 
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize



# API
app = Flask(__name__)
app.secret_key="chatty"
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
nltk.download('punkt')

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('sessions.html')

@app.route('/make-reply',methods=['GET','POST'])
def make_reply():
    message=request.form['message']

    # motbot script
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('dataintents.json', 'r') as json_data:
        dataintents = json.load(json_data)

    FILE = "data.pth" #file where new bag of words are stored and used when training the chatbot
    data = torch.load(FILE)
    #data input size
    input_size = data["input_size"]

    hidden_size = data["hidden_size"]
    #data output size
    output_size = data["output_size"]

    word_bag = data['word_bag']
    #loading all words and tags into data file inorder to have improve next session
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Motbot"
    print("Start chat! ")
    while True:
        lines = message
        if lines == "quit":
            break
        #tokenization
        lines = tokenize(lines)
        X = bag_of_words(lines, word_bag)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probability = torch.softmax(output, dim=1)
        prob = probability[0][predicted.item()]
        print(tag)
        if prob.item() > 0.65:
            for intent in dataintents ['dataintents']:
                if tag == intent["tag"]:
                    
                    return {"reply":random.choice(intent['responses'])}
        else:
            return {"reply":"Sorry, Can you ask more clearly as I did not understand..."}


if __name__ == '__main__':
    app.run(debug=True)
