"""Prior-free per-track estimates: MSD fits and Brownian displacement MLE, with explicit status."""
from ..data import Acquisition, Track
from ..io import AcquisitionParams, assert_contiguous_tracks, load_tracks, track_from_table
from .analysis import compute_msd
from .data import BrownianMLE, ClassicAnalysis, MLEOptions, MSDCurve, MSDFit, MSDOptions, TrackAnalysis
from .estimators import fit_anomalous_msd, fit_brownian_msd
from .likelihood import brownian_log_likelihood, fit_brownian_mle, nonbrownian_score
from .workflow import analyze_track, analyze_tracks

__all__ = [
    "Acquisition", "Track", "MSDOptions", "MSDCurve", "MSDFit", "TrackAnalysis",
    "ClassicAnalysis", "track_from_table", "compute_msd", "fit_brownian_msd",
    "fit_anomalous_msd", "analyze_track", "analyze_tracks", "MLEOptions", "BrownianMLE",
    "fit_brownian_mle", "brownian_log_likelihood", "nonbrownian_score",
]


def __getattr__(name):
    # Existing spt-pipeline/scripts keep their original behavior during migration.
    from ..legacy import classic
    try:
        value = getattr(classic, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import warnings
    warnings.warn(f"classic.{name} is legacy; use classic.analyze_track/analyze_tracks. "
                  "Legacy results have not been revalidated.", DeprecationWarning, stacklevel=2)
    return value
