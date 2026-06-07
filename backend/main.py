from fastapi import FastAPI
from backend.api import routes_root

app = FastAPI(title='UrbanEats Food Delivery ETA', description = "API Service for Real Time Inferencing",version="1.0.0")

app.include_router(routes_root.router)