from flask import Blueprint,render_template

bp = Blueprint("index",__name__,url_prefix="/")


@bp.route("/")
def get_index():
    return render_template("index.html")




@bp.route("/extract")
def demo():
    return render_template("extract.html")