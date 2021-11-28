import re
from flask import Flask, render_template, request
import model

app = Flask(__name__)


@app.get("/")
def hello():
    return render_template("home.html")
    
@app.route("/sub", methods = ['POST'])
def submit():
    if request.method == "POST":
        text = request.form["tarea"]
        res = model.predict_statement(text)
    return render_template("sub.html", v = res)


if __name__ == "__main__":
    app.run(debug=True)