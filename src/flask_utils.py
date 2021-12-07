import zipfile


def create_info_zip(var_dict, UPLOAD_DIR):
    # ZIP train files
    with zipfile.ZipFile(UPLOAD_DIR + "results/info.zip", "w") as file:
        file.write(UPLOAD_DIR + "results/valid_predictions.csv")
        file.write(UPLOAD_DIR + "results/train_valid_history.csv")
        file.write("./static/model/model.pkl")
        with open(UPLOAD_DIR + "results/params.txt", "w") as params:
            put_params(params, var_dict)
        file.write(UPLOAD_DIR + "results/params.txt")


def put_params(params, var_dict):
    params.write(f"Model type: " + "RandomForest" if var_dict["model_type"] == "1" else "GradientBoosting")
    params.write(f"N estimators {var_dict['model'].n_estimators}")
    params.write(f"Max depth: {var_dict['model'].max_depth}")
    params.write(f"Feature subsample size: {var_dict['model'].feature_subsample_size}")
    if var_dict["model_type"] == "2":
        params.write(f"Learning rate: {var_dict['learning_rate']}")
        params.write(f"Best train score: {var_dict['best_train_score']}")
        params.write(f"Training time: {var_dict['time_train']}")
        if var_dict["name_valid"] != "":
            params.write(f"Best valid accuracy: {var_dict['best_valid_score']}")