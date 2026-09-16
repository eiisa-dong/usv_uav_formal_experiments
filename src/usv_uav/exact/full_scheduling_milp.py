"""Legacy module retained only to explain the V2.1 rename.

The former implementation enumerated complete schedules and therefore was not
a full scheduling MILP. Import :mod:`tiny_schedule_column_milp` explicitly.
"""

from usv_uav.exact.tiny_schedule_column_milp import (
    TinyScheduleColumnMILP,
    TinyScheduleColumnResult,
)

__all__ = ["TinyScheduleColumnMILP", "TinyScheduleColumnResult"]
