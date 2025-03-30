# IronBox Logging Configuration

This document explains how logging is configured in the IronBox application and how to view and filter logs.

## Logging Configuration

The IronBox application uses Python's built-in logging module with the following configuration:

1. **In `ironbox/api/routes.py`**:
   - Root logger is configured with DEBUG level
   - Custom format with timestamp, module name, log level, and message
   - Specific loggers for different modules (uvicorn, fastapi, sqlalchemy, ironbox)

2. **In `ironbox/api/server.py`**:
   - Uvicorn server is configured to use debug log level

## Log Levels

The application uses standard Python logging levels:

- **DEBUG** (10): Detailed information, typically of interest only when diagnosing problems
- **INFO** (20): Confirmation that things are working as expected
- **WARNING** (30): Indication that something unexpected happened, or may happen in the near future
- **ERROR** (40): Due to a more serious problem, the software has not been able to perform some function
- **CRITICAL** (50): A serious error, indicating that the program itself may be unable to continue running

## Where Logs Are Stored

By default, logs are output to the console/terminal where the server is running. There is no separate log file configured.

## Viewing Logs

### Option 1: Run the Server Directly

When you run the server with:

```bash
python -m ironbox.api.server
```

All logs will be displayed in the terminal.

### Option 2: Use the Log Viewer Script

We've provided a log viewer script that allows you to run the server and filter logs:

```bash
cd /path/to/ironbox
./scripts/view_logs.py --follow
```

#### Script Options

- `--level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Minimum log level to display (default: DEBUG)
- `--module MODULE`: Filter logs by module name (e.g., 'ironbox.api')
- `--since SINCE`: Show logs since timestamp (format: YYYY-MM-DD HH:MM:SS)
- `--grep GREP`: Filter logs containing specific text
- `--tail TAIL`: Show only the last N lines (0 for all)
- `--follow, -f`: Follow the log output (like tail -f)

#### Examples

1. Show only ERROR and above logs:
   ```bash
   ./scripts/view_logs.py --follow --level ERROR
   ```

2. Filter logs from a specific module:
   ```bash
   ./scripts/view_logs.py --follow --module ironbox.api
   ```

3. Show logs containing a specific text:
   ```bash
   ./scripts/view_logs.py --follow --grep "cluster"
   ```

4. Combine filters:
   ```bash
   ./scripts/view_logs.py --follow --level WARNING --module ironbox.api --grep "error"
   ```

## Modifying Log Configuration

If you need to modify the logging configuration:

1. **Change Log Level**: Edit `ironbox/api/routes.py` to change the log level in the `logging.basicConfig()` call.

2. **Add File Logging**: Modify `ironbox/api/routes.py` to add a file handler:

   ```python
   # Add file handler
   file_handler = logging.FileHandler('ironbox.log')
   file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
   logging.getLogger().addHandler(file_handler)
   ```

3. **Configure Individual Loggers**: Adjust the log levels for specific modules in `ironbox/api/routes.py`:

   ```python
   logging.getLogger("module_name").setLevel(logging.LEVEL)
   ```
