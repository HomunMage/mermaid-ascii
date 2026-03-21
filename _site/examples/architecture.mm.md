graph TD
    subgraph Frontend
        GridView[Grid View]
        Timeline
        BoardView[Board View]
        LLMChat[LLM Chat]
    end
    subgraph Backend
        API[FastAPI]
    end
    Postgres[PostgreSQL]
    Minio["Minio\nblob store"]
    Claude["Claude API\ntool_use"]
    subgraph Worker
        Sync[Git Sync]
    end
    Frontend -->|HTTP| Backend
    Backend --> Postgres
    Backend --> Minio
    Backend --> Claude
    Worker -->|writes| Postgres
