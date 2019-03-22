
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/return")
def retpage():
    return render_template("return.html")
