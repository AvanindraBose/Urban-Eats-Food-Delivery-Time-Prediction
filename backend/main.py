from fastapi import FastAPI
from backend.api import routes_root,routes_auth,routes_predict
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from backend.core.rate_limiter import limiter


app = FastAPI(title='UrbanEats Food Delivery ETA', description = "API Service for Real Time Inferencing",version="1.0.0")

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

app.include_router(routes_root.router)
app.include_router(routes_auth.router)
app.include_router(routes_predict.router)