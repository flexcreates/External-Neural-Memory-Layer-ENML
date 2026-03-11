# Storage Subsystem

The storage subsystem handles JSON session persistence.

## Main File

| File | Purpose |
|---|---|
| `json_storage.py` | save, load, and list chat sessions |

## Session Layout

Sessions are stored under:

```text
memory/conversations/YYYY/MM/session_YYYYMMDD_HHMMSS.json
```

The storage layer supports:

- saving the current session
- loading a session by ID
- listing saved sessions across nested year/month directories

## Used By

- `chat.py`
- `web_server.py`
- `core/memory_manager.py`
