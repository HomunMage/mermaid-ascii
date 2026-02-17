subgraph "Svelte + Tailwind" {
  [Grid View]
  [Timeline]
  [Board View]
  [LLM Chat]
}

subgraph "FastAPI + SQLModel" {}

[PostgreSQL]
["Minio\n(blob store)"]
["Claude API\ntool_use"]

subgraph "Git Sync Worker (background)" {
  desc: "git fetch -> parse branches/tags -> openapi plugin -> sync DB"
}

("Svelte + Tailwind") --> ("FastAPI + SQLModel") { label: "HTTP" }
("FastAPI + SQLModel") --> [PostgreSQL]
("FastAPI + SQLModel") --> ["Minio\n(blob store)"]
("FastAPI + SQLModel") --> ["Claude API\ntool_use"]
("Git Sync Worker (background)") --> [PostgreSQL] { label: "writes" }
