import pandas as pd
from fastapi import APIRouter,Depends,Request
from backend.schema.request_schema import Requestschema
from fastapi.responses import JSONResponse
from backend.services.model_service import predict_time
from backend.core.dependencies import verify_token
from backend.logging_fastapi.logger_api import prediction_logger
from backend.core.rate_limiter import limiter

router = APIRouter(tags=["Predict"])

@router.post("/predict")
@limiter.limit("8/minute")
async def prediction(
    request:Request,
    data:Requestschema,
    status: dict = Depends(verify_token)
):
    try:
        prediction_logger.save_logs("Hit Prediction End Point",log_level="info")
        prediction_logger.save_logs(f"Prediction Served By:{status.get('payload').get('sub')}",log_level="info")
        
        pred_data = pd.DataFrame({
            'ID': data.ID,
            'Delivery_person_ID': data.Delivery_person_ID,
            'Delivery_person_Age': data.Delivery_person_Age,
            'Delivery_person_Ratings': data.Delivery_person_Ratings,
            'Restaurant_latitude': data.Restaurant_latitude,
            'Restaurant_longitude': data.Restaurant_longitude,
            'Delivery_location_latitude': data.Delivery_location_latitude,
            'Delivery_location_longitude': data.Delivery_location_longitude,
            'Order_Date': data.Order_Date,
            'Time_Orderd': data.Time_Orderd,
            'Time_Order_picked': data.Time_Order_picked,
            'Weatherconditions': data.Weatherconditions,
            'Road_traffic_density': data.Road_traffic_density,
            'Vehicle_condition': data.Vehicle_condition,
            'Type_of_order': data.Type_of_order,
            'Type_of_vehicle': data.Type_of_vehicle,
            'multiple_deliveries': data.multiple_deliveries,
            'Festival': data.Festival,
            'City': data.City
            },index=[0]
        )
        
        result = await predict_time(pred_data)
    except Exception as e:
        prediction_logger.save_logs(f"Error Occured During Prediction due to : {e}",log_level="error")
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "Failed to predict ETA",
                "retry": True
            }
        )
    
    else: 
        prediction_logger.save_logs("Prediction Successful",log_level="info")
        return JSONResponse (
            status_code=200,
            content = {
                "time": result.get("prediction"),
                "lower_limit": result.get("lower_limit"),
                "upper_limit": result.get("upper_limit"),
                "message": "ETA Prediction Successful",
                "retry": False
            }
        )

    
    