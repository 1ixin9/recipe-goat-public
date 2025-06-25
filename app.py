from flask import Flask, render_template, request
import webbrowser
from recipe_goat import parse_recipe
from threading import Timer

app = Flask(__name__)

user_input_data = ""
dropdown1_data = ""
dropdown2_data = ""


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/auto_submit', methods=['POST'])
def get_info():
    global user_input_data, dropdown1_data, dropdown2_data
    user_input_data = request.form.get('user_input', '')
    dropdown1_data = request.form.get('dropdown1', '')
    dropdown2_data = request.form.get('dropdown2', '')
    return f'Key Ingredient: {user_input_data}, Meal Type: {dropdown1_data}, Key Flavor: {dropdown2_data}'


@app.route('/result')
def result():
    user_input = request.args.get('user_input', '')
    dropdown1 = request.args.get('dropdown1', '')
    dropdown2 = request.args.get('dropdown2', '')

    info = parse_recipe(ingredient=user_input,
                        meal=dropdown1, flavor=dropdown2)

    name = info.get("Recipe Name", "")
    desc = info.get("Description", "")
    ingr = info.get("Ingredients", "")
    prep = info.get("Preparation", "")
    inst = info.get("Instructions", "")

    print(name, desc, ingr, prep, inst)

    return render_template('result.html', name=name, desc=desc, ingr=ingr, prep=prep, inst=inst)


def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')
    print("Browser opened")


if __name__ == '__main__':
    Timer(1, open_browser).start()
    print("Starting Flask app")
    app.run(debug=True)
