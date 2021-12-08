import pandas as pd
import numpy as np
import os
import pickle
import plotly.express as px
import zipfile
from collections import defaultdict


from flask import Flask, render_template, send_file, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired, InputRequired, Optional, ValidationError
from flask_bootstrap import Bootstrap

from ensembles import RandomForestRMSE, GradientBoostingRMSE
from flask_utils import create_info_zip

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hello'
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
UPLOAD_DIR = './src/static/datasets/'
MODEL_DIR = './src/static/model/'
Bootstrap(app)
target_column = 'price'
models = {'1': RandomForestRMSE, '2': GradientBoostingRMSE}
var_dict = defaultdict()

class DatasetForm(FlaskForm):
    train = FileField('Тренировочный датасет', validators=[DataRequired()])
    model_type = SelectField(
        'Model Type',
        choices=[(1, "RandomForest"), (2, "GradientBoosting")],
        validators=[InputRequired()]
    )
    valid = FileField('Валидационный датасет', validators=[])

    def validate_train(self, train):
        if train.data.filename[-4:] != ".csv":
            raise ValidationError(message="File must have .csv format")
        if not os.path.exists(UPLOAD_DIR):
            print(os.listdir())
            os.mkdir(UPLOAD_DIR[1:])
        train.data.save(UPLOAD_DIR + train.data.filename)
        train_ = pd.read_csv(UPLOAD_DIR + train.data.filename)
        self.train_columns = train_.columns
        if target_column not in train_.columns:
            os.remove(UPLOAD_DIR + train.data.filename)
            raise ValidationError(message="'target' must be in train columns")

    def validate_valid(self, valid):
        if len(valid.data.filename) != 0:
            if valid.data.filename[-4:] != ".csv":
                raise ValidationError(message="File must have .csv format")

            valid.data.save(UPLOAD_DIR + valid.data.filename)
            valid_ = pd.read_csv(UPLOAD_DIR + valid.data.filename)
            if set(self.train_columns) != set(valid_.columns):
                os.remove(UPLOAD_DIR + valid.data.filename)
                raise ValidationError(message="'Columns in valid dataset must match to train columns")

class ParamsFormRF(FlaskForm):
    n_estimators = StringField('n_estimators', validators=[])
    max_depth = StringField('max_depth', validators=[])
    feature_subsample_size = StringField('feature_subsample_size', validators=[])

    def validate_n_estimators(self, n_estimators):
        if len(n_estimators.data) != 0:
            error_message = "n_estimators parameter should be integer between 0 and 100 000."
            if len(n_estimators.data) > 5 or n_estimators.data[0] == '-':
                raise ValidationError(message=error_message)
            try:
                n_estimators.data = int(n_estimators.data)
            except ValueError:
                raise ValidationError(message=error_message)

            var_dict["n_estimators"] = n_estimators.data
        else:
            var_dict["n_estimators"] = None

    def validate_max_depth(self, max_depth):
        if len(max_depth.data) != 0:
            error_message = "max_depth parameter should be integer between 1 and 30 or -1."
            if len(max_depth.data) > 2 or max_depth.data[0] == '-' and max_depth.data[1:] != 1:
                raise ValidationError(message=error_message)
            try:
                max_depth.data = int(max_depth.data)
            except ValueError:
                raise ValidationError(message=error_message)
            var_dict["max_depth"] = max_depth.data
        else:
            var_dict["max_depth"] = None

    def validate_feature_subsample_size(self, feature_subsample_size):
        if len(feature_subsample_size.data) != 0:
            error_message = "feature_subsample_size parameter should be float in semi-interval (0, 1]."
            try:
                feature_subsample_size.data = float(feature_subsample_size.data)
            except ValueError:
                raise ValidationError(message=error_message)
            if feature_subsample_size.data > 1 or feature_subsample_size.data <= 0:
                raise ValidationError(message=error_message)

            var_dict["feature_subsample_size"] = feature_subsample_size.data
        else:
            var_dict["feature_subsample_size"] = None

class ParamsFormGB(ParamsFormRF):
    learning_rate = StringField('learning_rate', validators=[])

    def validate_learning_rate(self, learning_rate):
        if len(learning_rate.data) != 0:
            error_message = "learning_rate parameter must be positive float."
            try:
                learning_rate.data = float(learning_rate.data)
            except:
                raise ValidationError(message=error_message)

            if learning_rate.data <= 0:
                raise ValidationError(message=error_message)
            var_dict["learning_rate"] = learning_rate.data
        else:
            print("!!!!!!!!!")
            var_dict["learning_rate"] = None

class DatasetTestForm(FlaskForm):
    test = FileField('Тестовый датасет', validators=[DataRequired()])

    def validate_test(self, test):
            if test.data.filename[-4:] != ".csv":
                raise ValidationError(message="File must have .csv format")

            test.data.save(UPLOAD_DIR + test.data.filename)
            test_ = pd.read_csv(UPLOAD_DIR + test.data.filename)
            if set(var_dict["train_set"].drop(columns=target_column).columns) != set(test_.columns):
                os.remove(UPLOAD_DIR + test.data.filename)
                raise ValidationError(message="'Columns in test dataset must match to train columns except `target`")


@app.route('/', methods=["GET", "POST"])
def start():
    return render_template('index.html')


@app.route('/upload', methods=["GET", "POST"])
def upload():
    dataset_form = DatasetForm()
    if dataset_form.validate_on_submit():
        var_dict["model_type"] = dataset_form.model_type.data

        var_dict["train_set"] = pd.read_csv(UPLOAD_DIR + dataset_form.train.data.filename)
        var_dict["name_train"] = dataset_form.train.data.filename

        var_dict["name_valid"] = dataset_form.valid.data.filename
        if var_dict["name_valid"] != '':
            var_dict["valid_set"] = pd.read_csv(UPLOAD_DIR + dataset_form.valid.data.filename)

        return redirect(url_for('set_params'))

    if len(dataset_form.errors) != 0:
        return render_template("errors.html", errors=dataset_form.errors)

    return render_template('loading.html', form=dataset_form)


@app.route('/params', methods=['GET', 'POST'])
def set_params():
    if var_dict["model_type"] == '1':
        param_form = ParamsFormRF()
    else:
        param_form = ParamsFormGB()

    if param_form.validate_on_submit():
            return redirect(url_for('train_model'))

    if len(param_form.errors) != 0:
        return render_template("errors.html", errors=param_form.errors)

    if var_dict["model_type"] == '1':
        return render_template('rf_params.html', form=param_form)
    else:
        return render_template('gb_params.html', form=param_form)

@app.route('/training-model')
def train_model():
    # Train model
    var_dict["results"] = None
    if var_dict["model_type"] == "1":
        var_dict["model"] = models[var_dict["model_type"]](
            n_estimators=var_dict["n_estimators"],
            max_depth=var_dict["max_depth"],
            feature_subsample_size=var_dict["feature_subsample_size"],
        )
    else:
        var_dict["model"] = models[var_dict["model_type"]](
            n_estimators=var_dict["n_estimators"],
            max_depth=var_dict["max_depth"],
            feature_subsample_size=var_dict["feature_subsample_size"],
            learning_rate=var_dict["learning_rate"]
        )
    results = var_dict["model"].fit(
        np.array(var_dict["train_set"].drop(columns=[target_column, 'date'])),
        np.log(np.array(var_dict["train_set"][target_column])).reshape(-1),
        np.array(var_dict["valid_set"].drop(columns=[target_column, 'date'])) if var_dict["name_valid"] != '' else None,
        np.log(np.array(var_dict["valid_set"][target_column])).reshape(-1) if var_dict["name_valid"] != '' else None,
    )

    # Plot line chart
    train_group = np.repeat("Тренировка", len(results[0])).reshape(-1, 1)
    train_score = np.array(results[0]).reshape(-1, 1)
    timing = np.round(np.cumsum(results[1]), decimals=2).reshape(-1, 1)
    if len(results[-1]) != 0:
        timing = np.concatenate((timing, timing), axis=0)

    valid_score = np.array(results[-1]).reshape(-1, 1)
    valid_group = np.repeat("Валидация", len(results[-1])).reshape(-1, 1)

    df = pd.DataFrame(
        columns=['train_time', 'metric', "group"],
        data=np.concatenate(
            (
                timing,
                np.round(np.concatenate((train_score, valid_score), axis=0), decimals=3),
                np.concatenate((train_group, valid_group), axis=0)
            ),
            axis=1
        )
    )
    fig = px.line(
        df,
        x="train_time",
        y="metric",
        title='Зависимость метрики от времени',
        color="group",
        line_group="group",
        category_orders={"metric": [str(x) for x in reversed(np.sort(df.metric))]},
        labels=dict(train_time='Время тренировки, сек.', metric='Метрика', group='Стадия'),
        height=400
    )
    fig.update_layout(
        xaxis=dict(
            nticks=10,
        ),
        yaxis=dict(
            nticks=10,
        ),
    )
    # Save model
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    with open(MODEL_DIR + "model.pkl", "wb") as f:
        pickle.dump(var_dict["model"], f)

    if not os.path.exists(UPLOAD_DIR + "results/"):
        os.mkdir(UPLOAD_DIR + "results/")
    # Validation predictions
    if var_dict["name_valid"] != '':
        preds = var_dict["model"].predict(np.array(var_dict["valid_set"].drop(columns=[target_column, "date"])))
        predictions_pd = pd.DataFrame(data=preds, columns=['predictions'], index=var_dict["valid_set"].index)
        predictions_pd.to_csv(UPLOAD_DIR + "/results/valid_predictions.csv")

    # Save info
    var_dict["best_train_score"] = round(np.min(results[0]), ndigits=3)
    var_dict["time_train"] = round(np.sum(results[1][-1]), ndigits=3)
    if var_dict["name_valid"] != "":
        var_dict["best_valid_score"] = round(np.min(results[-1]), ndigits=3)
    df.to_csv(UPLOAD_DIR + "results/train_valid_history.csv")

    return render_template('learning.html', plot=fig.to_html())


@app.route('/training-mode/load-model', methods=['GET', 'POST'])
def load_model():
    return send_file("." + MODEL_DIR[5:] + "model.pkl", as_attachment=True)

@app.route('/training-model/test', methods=["GET", "POST"])
def test():
    dataset_form = DatasetTestForm()
    if dataset_form.validate_on_submit():
        test_set = pd.read_csv(UPLOAD_DIR + dataset_form.test.data.filename)

        preds = var_dict["model"].predict(test)
        preds = pd.DataFrame(index=test_set.index, data=preds)
        preds.to_csv(UPLOAD_DIR + "test_predictions.csv")
        return send_file("." + UPLOAD_DIR[5:] + "test_predictions.csv")

    if len(dataset_form.errors) != 0:
        return render_template("errors.html", errors=dataset_form.errors)

    return render_template('loading_test.html', form=dataset_form)


@app.route('/training-model/info', methods=["GET", "POST"])
def info():
    if "learning_rate" in var_dict:
        var_dict["learning_rate"] = var_dict["model"].learning_rate

    create_info_zip(var_dict, UPLOAD_DIR, MODEL_DIR)

    return render_template(
        'info.html',
        model_type="RandomForest" if var_dict["model_type"] == "1" else "GradientBoosting",
        n_estimators=var_dict["model"].n_estimators,
        max_depth=var_dict["model"].max_depth,
        feature_subsample_size=round(var_dict["model"].feature_subsample_size, ndigits=3),
        var_dict=var_dict
    )


@app.route('/training-model/load_info', methods=["GET", "POST"])
def load_info():
    return send_file("." + UPLOAD_DIR[5:] + 'results/info.zip', as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)