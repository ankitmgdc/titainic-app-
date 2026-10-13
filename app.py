from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# 1. मॉडल तैयार करना (ट्रेनिंग डेटा)
train_data = pd.DataFrame(
    {
        'Pclass': [3, 1, 3, 1, 2, 3, 1, 3],
        'Sex': [1, 0, 1, 0, 1, 0, 1, 0],  # 1 = Male, 0 = Female
        'Age': [22.0, 38.0, 26.0, 35.0, 54.0, 2.0, 28.0, 40.0],
        'Survived': [0, 1, 0, 1, 0, 1, 1, 1],
    }
)

X = train_data[['Pclass', 'Sex', 'Age']]
y = train_data['Survived']

model = RandomForestClassifier()
model.fit(X, y)


@app.route('/')
def home():
  return render_template('index.html')


# यह रूट फ्रंटएंड से डेटा लेगा और प्रेडिक्शन वापस करेगा
@app.route('/predict', methods=['POST'])
def predict():
  data = request.json
  pclass = int(data['pclass'])
  sex = int(data['sex'])
  age = float(data['age'])

  # बैकएंड गणना (मिलीसेकंड में)
  user_input = pd.DataFrame({'Pclass': [pclass], 'Sex': [sex], 'Age': [age]})
  prediction = int(model.predict(user_input)[0])

  # रिजल्ट JSON फॉर्मेट में फ्रंटएंड को भेजना
  return jsonify({'result': prediction})


if __name__ == '__main__':
  app.run(debug=True)

