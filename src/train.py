import argparse
from config import paths
from logger import get_logger, log_error
from data_models.data_validator import validate_data
from schema.data_schema import load_json_data_schema, save_schema
from utils import read_csv_in_directory, read_json_as_dict, set_seeds, split_train_val
from preprocessing_data.preprocessing_utils import save_pipeline , remove_all_model_artifacts
from preprocessing_data.pipeline import CategoricalTransformer , NumericTransformer , Merger , TargetEncoder , DataBalancer , FeatureSelection
from prediction.predictor_model import evaluate_predictor_model,save_predictor_model,train_predictor_model
from xai.explainer import fit_and_save_explainer
from hyperparameter_tuning.tuner import run_hyperparameter_tuning
import os

from sklearn.pipeline import Pipeline
import pandas as pd

logger = get_logger(task_name="train")


def run_training(
    input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path:str = paths.MODEL_CONFIG_FILE_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    train_dir:str = paths.TRAIN_DIR,
    explainer_config_file_path: str = paths.EXPLAINER_CONFIG_FILE_PATH,
    explainer_dir_path: str = paths.EXPLAINER_DIR_PATH,
    run_tuning: bool = False,
    ):
  
    try:
        
        # Making data preprocessing directory
        if not os.path.exists(paths.DATA_ARTIFACTS_DIR_PATH):
            os.makedirs(paths.DATA_ARTIFACTS_DIR_PATH)

        logger.info("Starting training...")
        # Removing previous files
        logger.info("Removing old files...")
        remove_all_model_artifacts()
        # load and save schema
        logger.info("Loading and saving schema...")
        data_schema = load_json_data_schema(input_schema_dir)
        save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)

        # load model config
        logger.info("Loading model config...")
        model_config = read_json_as_dict(model_config_file_path)

        set_seeds(seed_value=model_config["seed_value"])

        # load train data
        logger.info("Loading train data...")
        train_data = read_csv_in_directory(file_dir_path=train_dir)

        # validate the data
        logger.info("Validating train data...")
        validated_data = validate_data(
            data=train_data, data_schema=data_schema, is_train=True
        )
        
        # validated_data = validated_data.sample(frac = 1)

        # split train data into training and validation sets
        logger.info("Performing train/validation split...")
        train_split, val_split = split_train_val(
            validated_data, val_pct=model_config["validation_split"]
        )

        # custom logic for dropping irrelevant columns
        try:
            train_data.drop(labels=['month','day','day_week','hour','minute','a_region','state'],axis=1,inplace=True)
        except Exception as e:
            pass

        train_pipeline = Pipeline([
                                     ('CategoricalTransformer', CategoricalTransformer(data_schema.categorical_features,is_training=True)),
                                     ('NumericTransformer', NumericTransformer(data_schema.numeric_features,is_training=True)),
                                     ('Merger' , Merger())
                                  ])
        
        test_val_pipeline = Pipeline([
                                     ('CategoricalTransformer', CategoricalTransformer(data_schema.categorical_features,is_training=False)),
                                     ('NumericTransformer', NumericTransformer(data_schema.numeric_features,is_training=False)),
                                     ('Merger' , Merger())
                                  ])
        
        target_encoder_pipeline = Pipeline([('TargetEncoder',TargetEncoder(data_schema.target,data_schema.target_classes))])
        
        train_pipeline.fit(train_split)
        train_processed_data = train_pipeline.transform(train_split)
        train_processed_data = train_processed_data["processed_data"]
        save_pipeline(train_pipeline,"train_processing_pipeline")

        test_val_pipeline.fit(val_split)
        val_processed_data = test_val_pipeline.transform(val_split)
        val_processed_data = val_processed_data["processed_data"]
        save_pipeline(test_val_pipeline,"test_val_processing_pipeline")
        
        target_encoder_pipeline.fit(train_split)
        train_targets = target_encoder_pipeline.transform(train_split)
        val_targets = target_encoder_pipeline.transform(val_split)
        save_pipeline(target_encoder_pipeline,"target_encoder_pipeline")        

        # feature_selection_pipeline = Pipeline([('FeatureSelection',FeatureSelection())])
        # feature_selection_pipeline.fit({"X_train":train_processed_data,"Y_train":train_targets})
        # significant_features = feature_selection_pipeline.transform({"X_train":train_processed_data,"Y_train":train_targets})
        
        # if(len(significant_features)>=5):
        #     train_processed_data = train_processed_data[significant_features]
        #     val_processed_data = val_processed_data[significant_features]

        X_train = train_processed_data
        Y_train = train_targets
        X_val = val_processed_data
        Y_val = val_targets

        # data_balancer_pipeline = Pipeline([('DataBalancer',DataBalancer())])
        # data_balancer_pipeline.fit({"X_train":X_train,"Y_train":Y_train})
        # X_train , Y_train = data_balancer_pipeline.transform({"X_train":X_train,"Y_train":Y_train})

        logger.info("Training classifier...")
        default_hyperparameters = None

        # Making tuning default behaviour for now
        run_tuning = True
        
        if(run_tuning):
            logger.info("Tuning hyper paramters...")
            default_hyperparameters = run_hyperparameter_tuning(X_train,Y_train)
        else:
            default_hyperparameters = read_json_as_dict(
                default_hyperparameters_file_path
            )
        predictor = train_predictor_model(
            X_train, Y_train, default_hyperparameters
        )

        logger.info("Saving classifier...")
        save_predictor_model(predictor, predictor_dir_path)

        # calculate and print validation accuracy
        logger.info("Calculating accuracy on validation data...")
        val_accuracy = evaluate_predictor_model(
            predictor, X_val, Y_val
        )
        logger.info(f"Validation data accuracy: {val_accuracy}")

        logger.info("Fitting and saving explainer...")
        _ = fit_and_save_explainer(
            X_train, explainer_config_file_path, explainer_dir_path
        )
        
        logger.info("Training completed successfully")
   

    except Exception as exc:
       
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


def parse_arguments() -> argparse.Namespace:
    """Parse the command line argument that indicates if user wants to run
    hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Train a binary classification model.")
    parser.add_argument(
        "-t",
        "--tune",
        action="store_true",
        help=(
            "Run hyperparameter tuning before training the model. "
            + "If not set, use default hyperparameters.",
        ),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_training(run_tuning=args.tune)