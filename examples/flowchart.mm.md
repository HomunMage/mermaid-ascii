graph TD
    Start[Start] --> Decision{Decision}
    Decision -->|yes| ProcessA[Process A]
    Decision -->|no| ProcessB[Process B]
    ProcessA --> End[End]
    ProcessB --> End
