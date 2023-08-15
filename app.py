from flask import Flask, render_template, request
import json
import json_numpy
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from flask_socketio import SocketIO, emit, Namespace

msg = ""
glb = ""
add_acc = []
agg_param = ""
# plt_acc = []
count = 0
num_of_tl_clients = 2
recieved_data = 0
thershold = 0.0

app = Flask(__name__)
socketio = SocketIO(app)

model = tf.keras.models.load_model(
    "D:/Final Yr Pro/Code/FMLH-IDS/Federated/flr/Final/FedIoT/ae_binary"
)


@socketio.on("connect", namespace="/init")
def handle_init_connect():
    print("Client connected")
    data = json.dumps(
        {
            "weights": json_numpy.dumps(model.get_weights()),
            # "thershold": "0.15207744263617926",
            "thershold": "0.06919642008893982",
        }
    )
    socketio.emit("message", data, namespace="/init")


@socketio.on("disconnect", namespace="/init")
def handle_init_disconnect():
    print("Client disconnected")


@socketio.on("connect", namespace="/agg")
def handle_agg_connect():
    print("Client connected")


@socketio.on("disconnect", namespace="/agg")
def handle_agg_disconnect():
    print("Client disconnected")


@app.route("/")
def welcome():
    return render_template("index.html")


@app.route("/msg")
def msg_route():
    return msg


@app.route("/glb")
def msg_route2():
    return glb


@app.route("/data", methods=["POST"])
def receive_data():
    data = request.get_json()
    global msg, recieved_data, thershold, count, global_wght, add_acc, glb
    d1 = json.loads(data)
    model_weights = json_numpy.loads(d1["params"])
    # print(f"The Thershold is of type :{type(json.loads(d1['thershold']))}")
    thershold += json.loads(d1["thershold"])
    d1["params"] = str(model_weights)
    msg = d1
    acc_lst = np.array(json.loads(d1["accuracy_lst"]), dtype=float)

    if len(add_acc) == 0:
        add_acc = acc_lst
        print(f"The accuracy of the first client {add_acc}")
    else:
        if len(add_acc) != len(acc_lst):
            add_acc[-(len(acc_lst)) :] = sum(add_acc[-(len(acc_lst)) :]) / len(acc_lst)
            add_acc = add_acc[: -(len(acc_lst) - 1)]
        add_acc += acc_lst
        print(f"The accuracy list of the second client is {acc_lst}")
        print(f"The addition of them is {add_acc}\n\n")

    # print(type(model_weights))
    recieved_data += np.array(model_weights, dtype=object)
    # print("recieved_data", data)
    count += 1

    if num_of_tl_clients <= count:
        aggregated_param = recieved_data / num_of_tl_clients
        glb_thershold = thershold / 2
        global_wght = aggregated_param
        aggregated_param = aggregated_param.tolist()
        plt_acc = (add_acc / 2).tolist()
        print(f"The average of the acc_lst is {plt_acc}")
        glb = json.dumps(
            {
                "params": str(aggregated_param),
                "accuracy_lst": plt_acc,
                "thershold": glb_thershold,
            }
        )

        # print(glb)
        # print(type(glb))

        # print("\n\n\n\n\n\n\n")

        params = json.dumps(
            {
                "params": json_numpy.dumps(aggregated_param),
                # "accuracy_lst": plt_acc,
                "thershold": str(glb_thershold),
            }
        )

        # glb = params
        socketio.emit("agg_message", params, namespace="/agg")
        print("AGGREGATED MODEL SENT...........")
        recieved_data = 0
        add_acc = []
        thershold = 0
        count = 0
        print(f"EMPTY the acc {add_acc}")
    return "data recieved"


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
