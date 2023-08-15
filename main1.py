# clear && python ./flr/Final/FedIoT/main.py
import pandas as pd
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import time
import requests
import tensorflow as tf
from tensorflow.keras import layers
import json_numpy
import json
import joblib
import socketio
from sklearn.metrics import accuracy_score
from colorama import Fore
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import Model

accuracies = []
correct = 0
thershold = 0
sio = socketio.Client(True)
scaler = joblib.load(
    "D:/Final Yr Pro/Code/FMLH-IDS/Federated/flr/Final/FedIoT/minmax_scaler.pkl"
)


class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                # layers.Dense(95, activation="relu"),
                layers.Dense(48, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(48, activation="relu"),
                # layers.Dense(95, activation="relu"),
                layers.Dense(95, activation="tanh"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = AnomalyDetector()

# Compile the model

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, mode="min"
)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss="mae")


@sio.event
def connect():
    print("Connected to server")


@sio.event
def disconnect():
    print("Disconnected from server")


@sio.on("message", namespace="/init")
def handle_message(message):
    global model, thershold
    # print(f"The message {message}\n {json_numpy.loads(json.loads(message)['weights'])}")
    agg_r_init = json_numpy.loads(json.loads(message)["weights"])
    thershold = float(json.loads(message)["thershold"])
    # print(f"Thershold {thershold=}")
    model.set_weights(agg_r_init)
    print(
        Fore.LIGHTBLUE_EX
        + "MODEL Weights are INITIALIZED<<>>>>>>>>>>>>>>---------------\n"
    )


@sio.on("agg_message", namespace="/agg")  # Specify the namespace
def handle_agg_message(message):
    global model, thershold
    agg_json = json.loads(message)
    agg_r_init = json_numpy.loads(agg_json["params"])
    # print(f"THE NEW WEIGHT  IS \n\n\n\n {agg_r_init}\n")
    thershold = json.loads(agg_json["thershold"])
    # print(f"The global thershold is \n\n\n {thershold}")
    model.set_weights(agg_r_init)
    print(
        Fore.LIGHTGREEN_EX
        + "MODEL Weights are UPDATED<><>------------------------------\n"
    )


round = 1
y_test = []
acc = 0
params = 0


flask_url = "http://10.11.151.206:5000//data"
sio.connect(flask_url, wait=False, namespaces=["/agg", "/init"])


window = tk.Tk()
window.title("Client2")
window.geometry("800x600")

frame = tk.Frame(window)
frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

fig = Figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)

canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

label = tk.Label(frame, text="", justify=tk.LEFT, font=("Arial", 14), bg="white")
label.pack(side=tk.LEFT, padx=10)


df = pd.read_csv(
    "D:/Final Yr Pro/Code/FMLH-IDS/Federated/flr/Final/FedIoT/Kitsune_data/For_Fed.csv"
)

selected_col = [
    "MI_dir_L0.01_mean",
    "H_L0.01_mean",
    "MI_dir_L0.01_variance",
    "H_L0.01_variance",
    "MI_dir_L0.1_mean",
    "H_L0.1_mean",
    "MI_dir_L0.1_variance",
    "H_L0.1_variance",
    "MI_dir_L1_mean",
    "H_L1_mean",
    "H_L0.1_weight",
    "MI_dir_L0.1_weight",
    "MI_dir_L1_variance",
    "H_L1_variance",
    "HH_jit_L0.1_mean",
    "HH_jit_L0.01_mean",
    "MI_dir_L3_mean",
    "H_L3_mean",
    "HH_jit_L1_mean",
    "HH_jit_L3_mean",
    "HH_jit_L5_mean",
    "MI_dir_L5_mean",
    "H_L5_mean",
    "H_L1_weight",
    "MI_dir_L1_weight",
    "H_L3_variance",
    "MI_dir_L3_variance",
    "H_L5_variance",
    "MI_dir_L5_variance",
    "HH_L0.01_magnitude",
    "MI_dir_L0.01_weight",
    "H_L0.01_weight",
    "H_L3_weight",
    "MI_dir_L3_weight",
    "HH_L5_magnitude",
    "HH_L3_magnitude",
    "HH_L0.1_magnitude",
    "HH_L1_magnitude",
    "HH_L0.01_mean",
    "HpHp_L0.1_magnitude",
    "HpHp_L3_magnitude",
    "HpHp_L0.01_magnitude",
    "HpHp_L1_magnitude",
    "HpHp_L5_magnitude",
    "HH_L0.1_weight",
    "HH_jit_L0.1_weight",
    "HH_L0.1_mean",
    "HH_L1_mean",
    "HH_L5_mean",
    "HH_L3_mean",
    "HpHp_L0.1_mean",
    "HpHp_L5_mean",
    "HpHp_L1_mean",
    "HpHp_L3_mean",
    "HpHp_L0.01_mean",
    "MI_dir_L5_weight",
    "H_L5_weight",
    "HH_L0.01_std",
    "HH_L1_weight",
    "HH_jit_L1_weight",
    "HH_L0.01_radius",
    "HH_L0.1_std",
    "HH_jit_L3_weight",
    "HH_L3_weight",
    "HH_L0.1_radius",
    "HH_L5_weight",
    "HH_jit_L5_weight",
    "HH_L0.01_weight",
    "HH_jit_L0.01_weight",
    "HH_L5_std",
    "HH_L3_std",
    "HH_L1_std",
    "HpHp_L0.01_weight",
    "HpHp_L0.1_weight",
    "HH_jit_L0.01_variance",
    "HH_L0.01_pcc",
    "HH_jit_L0.1_variance",
    "HpHp_L0.01_std",
    "HH_L1_radius",
    "HH_L0.1_pcc",
    "HpHp_L0.1_std",
    "HpHp_L1_std",
    "HpHp_L1_weight",
    "HpHp_L3_weight",
    "HpHp_L5_weight",
    "HpHp_L3_std",
    "HH_L1_pcc",
    "HpHp_L5_std",
    "HH_L3_radius",
    "HH_L0.01_covariance",
    "HH_L0.1_covariance",
    "HH_L5_radius",
    "HH_jit_L1_variance",
    "HH_L3_pcc",
    "HH_L5_pcc",
    "is_attack",
    "Attack",
]


database = pd.DataFrame(columns=selected_col[:-1])
# database["isattack"] = None
# print(f"{len(database.columns)}")


for count, packet in enumerate(df.values):
    count += 1
    if count % 250 == 0:
        nrml_df = database[database["is_attack"] == 0]
        acc = accuracy_score(y_test, database["is_attack"])
        print(f"THe accuracy of this iteration is {acc=}")
        accuracies.append(acc)
        print(accuracies)
        if len(nrml_df) == 0:
            print(
                f"The Length of the normal data is {len(nrml_df)}.So posponding the training process"
            )
            time.sleep(1)
            # continue
        else:
            print(
                Fore.GREEN
                + "LOCAL TRAINING..................................................\n"
            )
            model.fit(
                nrml_df.iloc[:, :-1],
                nrml_df.iloc[:, :-1],
                epochs=50,
                batch_size=64,
                validation_data=(database.iloc[:, :-1], database.iloc[:, :-1]),
                shuffle=True,
                callbacks=[early_stop],
            )
            reconstructions = model.predict(nrml_df.iloc[:, :-1])
            train_loss = tf.keras.losses.mae(reconstructions, nrml_df.iloc[:, :-1])
            threshold = np.mean(train_loss) + 3 * np.std(train_loss)
            print(f"The thershold of the iteration is {threshold}")

        headers = {"Content-Type": "application/json"}

        # print(nrml_df)
        # break

        y_test = []

        # Define the model architecture

        params = json_numpy.dumps(model.get_weights())
        database = database[0:0]
        data = {
            "addrs": "Client2",
            "params": params,
            "acc": acc,
            "accuracy_lst": json.dumps(accuracies),
            "thershold": json.dumps(threshold),
            "size": 100,
        }
        response = requests.post(flask_url, json=json.dumps(data), headers=headers)
        if response.status_code == 200:
            print("Data sent successfully")
            print(
                Fore.MAGENTA
                + "LOCAL PARAMETER SEND >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
            )

        else:
            print("Retrying to senda data...................!")
            while response.status_code != 200:
                time.sleep(2)
                # time.sleep(8)
                response = requests.post(
                    flask_url, json=json.dumps(data), headers=headers
                )
                print(response.status_code)

    else:
        # print(packet)

        # SCALING
        packet = pd.DataFrame(packet).T
        # print(packet)
        # print(packet.shape)
        grd_trth = int(packet.iloc[:, -2:-1].values)
        packet = scaler.transform(packet.iloc[:, :-2])
        packet = pd.DataFrame(packet, columns=df.columns[:-2])[
            selected_col[:-2]
        ].to_numpy()
        print("PREPROCESSED")
        # TKINTER
        ax.clear()
        ax.plot(packet.reshape(-1, 1))
        ax.set_xlabel("Features")
        ax.set_ylabel("Readings")
        canvas.draw()
        window.update()
        # features = np.array(packet[:-1]).reshape(1, -1)
        y_test.append(grd_trth)
        # gr_th = packet.pop()
        # print(
        #     Fore.YELLOW
        #     + "LOCAL TESTING......................................................\n"
        # )

        reconstruction = model.predict(packet)
        loss = tf.keras.losses.mae(reconstruction, packet)
        ypre = int(tf.math.less(loss, thershold).numpy())
        packet = packet[0].tolist()
        # print(packet)
        print(f"{ypre=} , grd_th = {grd_trth},  {database.shape=}")

        if ypre == grd_trth:
            correct += 1
            if ypre == 0:
                label.config(text="Non-Malicious", fg="light green")
                label.update()
                packet.append(0)
                # print(f"The packet {packet=} \n {len(packet)}")
                database.loc[len(database)] = packet
            else:
                label.config(text="Malicious", fg="light green")
                label.update()
                packet.append(1)
                # print(f"The packet {packet=} \n {len(packet)}")
                database.loc[len(database)] = packet
        else:
            if ypre == 0:
                label.config(text="Non-Malicious", fg="red")
                label.update()
                packet.append(0)
                # print(f"The packet {packet=} \n {len(packet)}")
                database.loc[len(database)] = packet
            else:
                label.config(text="Malicious", fg="red")
                label.update()
                packet.append(1)
                # print(f"The packet {packet=} \n {len(packet)}")
                database.loc[len(database)] = packet

        # time.sleep(1)
        # window.after(10)
        print("---------------------------APPEND DATA-----------------------------")

window.mainloop()
