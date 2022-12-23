__author__ = ["Joren Retel"]
__email__ = ["joren.retel@bayer.com"]

try:
    from nucleotides._version import version as __version__
except ImportError:
    # this protects against an edge case where a user tries to import
    # the module without inquit()
    # stalling it, by adding it manually to
    # sys.path or trying to run it directly from the git checkout.
    __version__ = "not-installed"

from nucleotides.settings import Settings as _Settings

config = _Settings()
