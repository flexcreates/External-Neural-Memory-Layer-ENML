# Tools Module

The tools module provides sandboxed utilities for file system operations.

## Files

### `file_tool.py` — FileTool

A sandboxed file I/O interface that restricts all operations to directories listed in the `ALLOWED_PATHS` environment variable.

**Security Model:**
- All paths are resolved to absolute paths before validation
- Path traversal attacks (e.g., `../../etc/passwd`) are blocked by `Path.resolve()`
- Default allowed paths: `ENML_ROOT`, `~/Projects`, `~/Research`

**Methods:**
| Method | Description |
|---|---|
| `validate_path(path)` | Returns `True` if path is within allowed directories |
| `read_file(path)` | Reads file content (UTF-8) after validation |
| `write_file(path, content)` | Writes content to file, creating parent dirs as needed |
| `list_dir(path)` | Lists directory contents (filenames only) |

**Configuration:**
Set `ALLOWED_PATHS` in `.env` as a comma-separated list of absolute paths:
```env
ALLOWED_PATHS=/home/flex/Projects,/home/flex/Documents,/home/flex/Research
```

**Usage:**
```python
from tools.file_tool import FileTool

tool = FileTool()
if tool.validate_path("/home/flex/Projects/main.py"):
    content = tool.read_file("/home/flex/Projects/main.py")
```
