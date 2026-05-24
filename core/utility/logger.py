import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def success(self, message, *args, **kws):
    """Log a message with the custom SUCCESS level."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kws)


# Register SUCCESS on the standard Logger class.
logging.Logger.success = success

custom_theme = Theme({
    "logging.level.info": "blue",
    "logging.level.warning": "yellow",
    "logging.level.error": "red",
    "logging.level.success": "bold green",
})

console = Console(theme=custom_theme)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
    ]
)

logger = logging.getLogger("pipeline")
