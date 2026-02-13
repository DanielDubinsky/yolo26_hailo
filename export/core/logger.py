import logging
import sys
from pathlib import Path
from typing import Optional

class StreamRedirector:
    """Redirects stdout/stderr to both console and a log file."""
    
    def __init__(self, log_file: Path, stream_name: str = "stdout"):
        self.log_file = log_file
        self.stream_name = stream_name
        self.terminal = sys.stdout if stream_name == "stdout" else sys.stderr
        self.log = None
        self._last_line_was_progress = False
        
    def __enter__(self):
        self.log = open(self.log_file, 'a', buffering=1)  # Line buffered
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            self.log.close()
        
    def write(self, message):
        # Always write to terminal
        self.terminal.write(message)
        self.terminal.flush()
        
        # Filter progress bars from log file
        # Progress bars typically use \r (carriage return) to update in place
        if self.log:
            # Check if this is a progress bar update (contains \r but not \n)
            is_progress_update = '\r' in message and '\n' not in message
            
            # If the previous line was a progress update and this is a newline,
            # write it to preserve formatting
            if self._last_line_was_progress and message == '\n':
                self.log.write(message)
                self.log.flush()
                self._last_line_was_progress = False
            # Only write non-progress lines to the log
            elif not is_progress_update:
                # If this line contains \r followed by \n, it's the final state
                # Strip the \r characters for cleaner logs
                clean_message = message.replace('\r\n', '\n').replace('\r', '')
                if clean_message:  # Only write if there's actual content
                    self.log.write(clean_message)
                    self.log.flush()
                self._last_line_was_progress = False
            else:
                self._last_line_was_progress = True


            
    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()
    
    def isatty(self):
        """Check if the underlying terminal is a TTY."""
        return self.terminal.isatty()

def setup_logger(name: str, log_file: Path = None, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that writes to console and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
        
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def redirect_output_to_log(log_file: Path):
    """
    Redirects stdout and stderr to both console and log file.
    Returns the original stdout and stderr for restoration.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create redirectors
    stdout_redirector = StreamRedirector(log_file, "stdout")
    stderr_redirector = StreamRedirector(log_file, "stderr")
    
    # Enter context managers
    stdout_redirector.__enter__()
    stderr_redirector.__enter__()
    
    # Replace sys.stdout and sys.stderr
    sys.stdout = stdout_redirector
    sys.stderr = stderr_redirector
    
    return original_stdout, original_stderr, stdout_redirector, stderr_redirector

def restore_output(original_stdout, original_stderr, stdout_redirector, stderr_redirector):
    """Restores original stdout and stderr."""
    stdout_redirector.__exit__(None, None, None)
    stderr_redirector.__exit__(None, None, None)
    sys.stdout = original_stdout
    sys.stderr = original_stderr
