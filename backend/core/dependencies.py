import pandas as pd
import numpy as np
from backend.loaders.model_pipeline_loader import load_artifacts

async def get_artifact():
    return await load_artifacts()

        