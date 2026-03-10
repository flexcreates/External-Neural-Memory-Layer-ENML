# Storage Subsystem

The storage subsystem handles JSON-based session persistence for conversation history.

## Files

### `json_storage.py` — JSONStorage

Manages conversation session files with a date-based directory structure:

```
memory/conversations/
├── 2026/
│   ├── 01/
│   │   └── session_20260115_143022.json
│   └── 02/
│       ├── session_20260222_184853.json
│       └── session_20260222_185354.json
```

**Session File Format:**
```json
{
  "session_id": "session_20260222_184853",
  "timestamp": "2026-02-22T18:48:53",
  "messages": [
    {"role": "user", "content": "my name is Flex"},
    {"role": "assistant", "content": "Nice to meet you, Flex!"}
  ]
}
```

**Key Methods:**
| Method | Description |
|---|---|
| `save_session(id, messages)` | Saves to `YYYY/MM/{id}.json` |
| `load_session(id)` | Searches recursively (flat path + subdirectories) |
| `list_sessions()` | Returns all session IDs across all subdirectories |

**Resuming Sessions:**
```bash
python3 chat.py --session session_20260222_184853
```
