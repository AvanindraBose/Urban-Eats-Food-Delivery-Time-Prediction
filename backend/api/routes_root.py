from fastapi import APIRouter

router = APIRouter(tags=['Root'])

@router.get("/")
def root():
    return {
        "msg" : "Welcome to the route of Urban Eats Food Delivery ETA Prediction App."
    }
