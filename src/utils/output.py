"""Optional output/logging adapter that preserves current defaults."""

import logging
import sys
import warnings
from typing import Any, Mapping, Optional

LOGGER_NAME = "D4CMPP2"
logger = logging.getLogger(LOGGER_NAME)
if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
    logger.addHandler(logging.NullHandler())


class OutputAdapter:
    """Route opt-in diagnostics without changing default stdout/warnings."""

    def __init__(
        self,
        *,
        verbose: bool = True,
        stream=None,
        error_stream=None,
        diagnostic_logger: Optional[logging.Logger] = None,
    ) -> None:
        if not isinstance(verbose, bool):
            raise TypeError(
                f"verbose must be a bool, got {type(verbose).__name__}: {verbose!r}."
            )
        self.verbose = verbose
        # Keep None so redirect_stdout() applied after construction still works.
        self.stream = stream
        self.error_stream = error_stream if error_stream is not None else stream
        self.logger = diagnostic_logger if diagnostic_logger is not None else logger

    def info(self, *values: Any, **kwargs: Any) -> None:
        """Match the existing print-based user output."""

        if self.verbose:
            print(*values, file=self.stream, **kwargs)

    def error(self, *values: Any, **kwargs: Any) -> None:
        """Write essential failure information regardless of verbosity."""

        print(*values, file=self.error_stream, **kwargs)

    def always(self, *values: Any, **kwargs: Any) -> None:
        """Write an essential completion summary regardless of verbosity."""

        print(*values, file=self.stream, **kwargs)

    def progress(self, iterable, **kwargs):
        """Wrap an iterable in tqdm while respecting the verbosity policy."""

        from tqdm import tqdm

        kwargs.setdefault("disable", not self.verbose)
        return tqdm(iterable, **kwargs)

    def warning(
        self,
        message: str,
        category=UserWarning,
        *,
        stacklevel: int = 2,
    ) -> None:
        """Keep actionable fallback/deprecation messages visible."""

        warnings.warn(message, category, stacklevel=stacklevel)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Emit opt-in developer diagnostics through the package logger."""

        self.logger.debug(message, *args, **kwargs)


def get_output(config: Optional[Mapping[str, Any]] = None) -> OutputAdapter:
    """Build the standard output adapter for a runtime configuration."""

    verbose = True if config is None else config.get("verbose", True)
    return OutputAdapter(verbose=verbose)
