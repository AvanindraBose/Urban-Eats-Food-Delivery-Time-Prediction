Project Architecture
====================

Below is a detailed architecture diagram describing the components, data flow, and CI interactions for the Urban Eats Delivery Time Prediction project.

```mermaid
flowchart LR

  subgraph Dev[Developer / Local]
    AIDE[IDE / Notebooks]
    LocalEnv[Virtualenv / uv / Tools]
    AIDE -->|Edit code / notebooks| BackendSrc(backend, src, services)
    AIDE -->|Run training| TrainScript[src/models/train_model.py]
    LocalEnv -->|Run API| BackendSrc
    LocalEnv -->|Run tests| Tests[tests/]
  end

  subgraph DataAndDVC[Data & Pipelines]
    Raw[data/raw]
    Interim[data/interim]
    Processed[data/processed]
    DVC[dvc.yaml / dvc.lock]
    Raw -->|preprocess| Interim -->|transform| Processed
    DVC -.-> Raw
    DVC -.-> Processed
  end

  subgraph Training[Training & Evaluation]
    Trainer[src/models/train_model.py]
    Preprocessor["models/preprocessor.joblib"]
    ModelFiles["models/*.joblib"]
    Evaluator[src/models/evaluate_model.py]
    Trainer -->|save artifacts| ModelFiles
    Trainer -->|save transformer| Preprocessor
    Evaluator -->|log metrics & model| MLflow[MLflow / DagsHub]
    ModelFiles --> MLflow
  end

  subgraph ModelRegistry[Model Registry / Storage]
    MLflow
    ArtifactStore[S3 / DVC remote]
    MLflow -->|register model| ModelRegistry[Registered Model: delivery_time_pred_model_pipe]
    ArtifactStore -->|store large files| ModelFiles
  end

  subgraph Serving[API / Runtime]
    Backend[FastAPI - backend/main.py]
    Routes["backend/api/*"]
    Loaders[backend/loaders/model_pipeline_loader.py]
    Services[backend/services/model_service.py]
    Backend --> Routes
    Routes -->|calls| Services
    Services -->|loads model via| Loaders
    Loaders -->|fetch from| MLflow
    Backend -->|returns| Clients[Clients / Frontend / HTTP]
  end

  subgraph CI[CI / CD]
    GitHub[.github/workflows/ci-cd.yaml]
    CI_Lint[Linter & Workflow checks]
    CI_Tests[Test runner (pytest)]
    CI_DVC[DVC pull & repro]
    CI_Build[Docker build]
    CI_Push[ECR / DockerHub]
    GitHub --> CI_Lint
    GitHub --> CI_Tests
    GitHub --> CI_DVC
    GitHub --> CI_Build --> CI_Push
    CI_DVC --> MLflow
  end

  subgraph Infra[Infrastructure]
    ECR[ECR Registry]
    EKS_or_ECS[Cluster / Service]
    ECR --> EKS_or_ECS
    MLflow -->|tracked runs| DagsHubRemote[DagsHub / MLflow Server]
  end

  %% cross-links
  Processed --> Trainer
  ModelFiles --> Loaders
  MLflow --> Loaders
  CI_Push --> ECR
  CI_Tests -->|artifact logs| Logs[logs/]
  Backend --> Logs

  %% notes
  classDef infra fill:#f9f,stroke:#333,stroke-width:1px;
```

Legend / notes

- Developer: code and experiments are authored locally (notebooks and `src`).
- Data: Raw -> Interim -> Processed; DVC tracks pipeline steps in `dvc.yaml`.
- Training: `src/models/train_model.py` creates model artifacts saved under `models/` and uploaded to MLflow/DagsHub.
- Serving: FastAPI (`backend/`) exposes `/predict`; `backend/loaders/model_pipeline_loader.py` loads the production model from MLflow.
- CI: GitHub Actions runs lint/tests, DVC pulls, and builds/pushes images to ECR; promotion steps run after successful main branch pipelines.

Suggested next additions (optional)

- An image export of this Mermaid diagram (`svg`/`png`) for documentation sites.
- Component-level diagrams (sequence diagrams) for: training run, prediction request flow, CI promotion flow.
