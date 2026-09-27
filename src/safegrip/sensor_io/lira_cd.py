from __future__ import annotations

import pandas as pd
from .prepared import prepared_friction_frame_to_record


def lira_cd_frame_to_record(frame: pd.DataFrame, *, sequence_id: str = "lira-cd"):
    return prepared_friction_frame_to_record(frame, sequence_id=sequence_id, source_domain="lira_cd")
