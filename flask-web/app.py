from flask import Flask
from blueprints.index import bp as bp_index
from blueprints.model import bp as bp_model
from biobert_casrel_model.model import CasRel

app = Flask(__name__)
app.register_blueprint(bp_index)
app.register_blueprint(bp_model)



if __name__ == '__main__':
    app.run(host="0.0.0.0",port=80,debug=True)
