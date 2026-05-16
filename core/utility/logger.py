import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Define a custom SUCCESS log level
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def success(self, message, *args, **kws):
    """Log a message with the custom SUCCESS level."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kws)

# Add the custom level native to python's logging
logging.Logger.success = success

# Configure custom colors for log levels for RichHandler
custom_theme = Theme({
    "logging.level.info": "blue",
    "logging.level.warning": "yellow",
    "logging.level.error": "red",
    "logging.level.success": "bold green",
})

# Initialize Rich Console
console = Console(theme=custom_theme)

# Configure the standard Python logger using RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", # RichHandler takes care of the date, time, and level formatting
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console, 
            rich_tracebacks=True, 
            markup=True,
            show_time=True,
            show_path=False # Set to True if you want script names/line numbers on the right
        )
    ]
)

# Export the centralized logger and console
logger = logging.getLogger("pipeline")
