from flask import Flask,render_template,request
import pickle

cv = pickle.load(open("model/vector.pk1", "rb"))
clf = pickle.load(open("model/model.pk1", "rb"))

sample = """"
Hi Rahul 

Yesterday I saw you presentation. It was quite nice. 
I would like to offer you a job at our company

Regards Amit
Tech Lead
Amazon
"""


app = Flask(__name__)

@app.route('/')
def index():
    #vectorization
   # result  = cv.transform([sample]).toarray()
    #predict
   # pred = clf.predict(result)
   # print(pred)
    return render_template("index.html")

@app.route('/predict', methods=['post'])
def predict():
    UserInput = request.form.get('email')
    result  = cv.transform([sample]).toarray()
    # predict
    pred = clf.predict(result)
    print(pred)
    return UserInput


if __name__ == "__main__":
    app.run(debug=True)
