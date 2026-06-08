from fastapi import FastAPI
from backend.api import routes_root,routes_auth,routes_predict

app = FastAPI(title='UrbanEats Food Delivery ETA', description = "API Service for Real Time Inferencing",version="1.0.0")

app.include_router(routes_root.router)
app.include_router(routes_auth.router)
app.include_router(routes_predict.router)