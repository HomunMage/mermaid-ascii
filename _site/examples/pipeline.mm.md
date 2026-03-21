flowchart LR
    Source --> Build --> Test --> Deploy
    Build --> Lint
    Lint --> Test
