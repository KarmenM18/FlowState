# bridge_server.py
from flask import Flask
import threading, subprocess

app = Flask(__name__)
process = None

@app.route('/start')
def start():
    global process
    if process is None:
        process = subprocess.Popen(["python", "sensor_bridge.py"])
        return "Started sensor bridge"
    return "Already running"

@app.route('/stop')
def stop():
    global process
    if process:
        process.terminate()
        process = None
        return "Stopped sensor bridge"
    return "Not running"

if __name__ == "__main__":
    app.run(port=5000)
