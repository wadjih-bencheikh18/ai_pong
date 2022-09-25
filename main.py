
import os
import pickle
import neat
from flask_cors import CORS
from flask import Flask, redirect, url_for, request, render_template

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config.txt")

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

with open("best.pickle", "rb") as f:
    winner = pickle.load(f)


def test_ai(paddle, ball):
    [px, py] = paddle
    [bx, by] = ball
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    output = net.activate(
        (py, by, abs(px - bx)))
    decision = output.index(max(output))
    return decision


# if __name__ == "__main__":
#     print(test_ai([300, 300], [100, 100]))

app = Flask(__name__)
CORS(app)


@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getPos', methods=['POST'])
def getPos():
    data = request.get_json()
    ball = data["params"]['ball']
    paddle = data["params"]["paddle"]
    return str(test_ai(paddle, ball))


if __name__ == '__main__':
    app.run(debug=True)
