"""
This script contains utility functions/classes that are used in serve.py
"""
import uuid
from typing import Any, Dict, Tuple
import joblib
import pandas as pd
from config import paths
from data_models.data_validator import validate_data
from logger import get_logger, log_error
from prediction.predictor_model import load_predictor_model, predict_with_model
from schema.data_schema import load_saved_schema
from utils import read_json_as_dict
from xai.explainer import load_explainer
from preprocessing_data.preprocessing_utils import load_pipeline , load_correlated_features
from predict import create_predictions_dataframe

logger = get_logger(task_name="serve")




def generate_unique_request_id():
    """Generates unique alphanumeric id"""
    return uuid.uuid4().hex[:10]


async def transform_req_data_and_make_predictions(dataframe,request_id):
    logger.info("Transforming data sample(s)...")
    test_val_pipeline = load_pipeline("test_val_processing_pipeline")
    transformed_data = test_val_pipeline.transform(dataframe)
    transformed_data = transformed_data["processed_data"]
    
   

    # check for correlated features to be selected at prediction
    correlated_features = load_correlated_features()
    if(len(correlated_features)>=1):
        transformed_data = transformed_data[correlated_features]
    

    logger.info("Making predictions...")
    predictions_arr = predict_with_model(
       load_predictor_model(paths.PREDICTOR_DIR_PATH), transformed_data, return_probs=True
    )
    
    logger.info("Converting predictions array into dataframe...")

    saved_schema = load_saved_schema(paths.SAVED_SCHEMA_DIR_PATH)
    model_config = read_json_as_dict(paths.MODEL_CONFIG_FILE_PATH)

    predictions_df = create_predictions_dataframe(
        predictions_arr,
        saved_schema.target_classes,
        model_config["prediction_field_name"],
        dataframe[saved_schema.id],
        saved_schema.id,
        return_probs=True
    )
    
    logger.info("Converting predictions dataframe into response dictionary...")
    predictions_response = create_predictions_response(
        predictions_df, saved_schema, request_id
    )
    return transformed_data, predictions_response
    


def create_predictions_response(
    predictions_df: pd.DataFrame, data_schema: Any, request_id: str
) -> Dict[str, Any]:
    """
    Convert the predictions DataFrame to a response dictionary in required format.

    Args:
        transformed_data (pd.DataFrame): The transfomed input data for prediction.
        data_schema (Any): An instance of the BinaryClassificationSchema.
        request_id (str): Unique request id for logging and tracking

    Returns:
        dict: The response data in a dictionary.
    """
    class_names = data_schema.target_classes
    # find predicted class which has the highest probability
    predictions_df["__predicted_class"] = predictions_df[class_names].idxmax(axis=1)
    sample_predictions = []
    for sample in predictions_df.to_dict(orient="records"):
        sample_predictions.append(
            {
                "sampleId": sample[data_schema.id],
                "predictedClass": str(sample["__predicted_class"]),
                "predictedProbabilities": [
                    round(sample[class_names[0]], 5),
                    round(sample[class_names[1]], 5),
                ],
            }
        )
    predictions_response = {
        "status": "success",
        "message": "",
        "timestamp": pd.Timestamp.now().isoformat(),
        "requestId": request_id,
        "targetClasses": class_names,
        "targetDescription": data_schema.target_description,
        "predictions": sample_predictions,
    }
    return predictions_response


def combine_predictions_response_with_explanations(
    predictions_response: dict, explanations: dict
) -> dict:
    """
    Combine the predictions response with explanations.

    Inserts explanations for each sample into the respective prediction dictionary
    for the sample.

    Args:
        predictions_response (dict): The response data in a dictionary.
        explanations (dict): The explanations for the predictions.
    """
    for pred, exp in zip(
        predictions_response["predictions"], explanations["explanations"]
    ):
        pred["explanation"] = exp
    predictions_response["explanationMethod"] = explanations["explanation_method"]
    return predictions_response


# def load_predictor():
#     try:
#         model = joblib.load( joblib.load(paths.PREDICTOR_DIR_PATH)+"/predictor.joblib")
#         return model
#     except Exception as e :
#         raise f"Error while loading predictor : {e}"