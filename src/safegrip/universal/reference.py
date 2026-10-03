from __future__ import annotations

"""Parameter-free physical references for UniversalSafeGrip queries.

The reference operator is semantic rather than dataset-specific: rules are
registered by physical target type and operate only on canonical quantity,
axis, location and unit metadata.  The learned decoder always estimates the
innovation around the returned reference.
"""

import torch

from .batching import UniversalSensorBatch
from .ontology import (
    AXIS_TO_ID,
    LOCATION_TO_ID,
    QUANTITY_TO_ID,
    canonical_name,
    ontology_id,
)

STANDARD_GRAVITY = 9.80665


class PhysicalReferenceOperator:
    """Map a sensor batch and semantic query to a physical reference value.

    No trainable parameters or dataset identifiers are used.  A registry maps
    query target types to physically defined operators:

    * state_component -> latest matching observed state;
    * road_friction   -> horizontal specific-force demand ||a_xy|| / g;
    * localization    -> planar dead-reckoned endpoint displacement from measured body speed and yaw rate (straight-line fallback).

    Unsupported/unobservable queries receive the neutral reference zero, so the
    original UniversalSafeGrip decoder remains unchanged for those queries.
    """

    def __init__(self):
        self._rules = {
            "state_component": self._state_reference,
            "road_friction": self._friction_reference,
            "localization": self._localization_reference,
        }

    @staticmethod
    def _zero(batch: UniversalSensorBatch, query):
        b = batch.features.shape[0]
        return (
            batch.features.new_zeros((b,)),
            torch.zeros((b,), dtype=torch.bool, device=batch.features.device),
        )

    @staticmethod
    def _latest(batch: UniversalSensorBatch, match: torch.Tensor, values: torch.Tensor):
        neg_inf = torch.full_like(batch.times_sec, float("-inf"))
        latest_time = torch.where(match, batch.times_sec, neg_inf).max(dim=1).values
        latest = match & torch.isclose(
            batch.times_sec, latest_time.unsqueeze(1), rtol=0.0, atol=1e-7
        )
        count = latest.sum(dim=1)
        ok = count > 0
        selected = torch.where(latest, values, torch.zeros_like(values))
        val = selected.sum(dim=1) / count.clamp_min(1).to(values.dtype)
        return torch.where(ok, val, torch.zeros_like(val)), ok

    def _state_reference(self, batch: UniversalSensorBatch, query):
        if batch.last_values is None:
            return self._zero(batch, query)
        qid, aid, lid, _ = query.ids()
        match = (
            batch.token_mask.bool()
            & (batch.channel_ids >= 0)
            & (batch.quantity_ids == int(qid))
            & (batch.axis_ids == int(aid))
            & (batch.location_ids == int(lid))
        )
        return self._latest(batch, match, batch.last_values)

    def _friction_reference(self, batch: UniversalSensorBatch, query):
        """Horizontal specific-force demand used as a mechanics reference.

        This is deliberately *not* claimed to be the road-friction coefficient
        or a deterministic lower bound.  It is a dimensionless, physically
        meaningful excitation reference around which the decoder learns the
        remaining road-friction innovation.
        """
        if batch.rms_values is None:
            return self._zero(batch, query)
        acc_id = ontology_id("acceleration", QUANTITY_TO_ID)
        body_id = ontology_id("vehicle_body", LOCATION_TO_ID)
        axis_ids = (
            ontology_id("longitudinal", AXIS_TO_ID),
            ontology_id("lateral", AXIS_TO_ID),
        )
        components = []
        observed = []
        for axis_id in axis_ids:
            match = (
                batch.token_mask.bool()
                & (batch.channel_ids >= 0)
                & (batch.quantity_ids == acc_id)
                & (batch.axis_ids == axis_id)
                & (batch.location_ids == body_id)
            )
            value, ok = self._latest(batch, match, batch.rms_values)
            components.append(value)
            observed.append(ok)
        sq = sum(value.square() for value in components)
        any_observed = torch.stack(observed, dim=0).any(dim=0)
        ref = torch.sqrt(sq.clamp_min(0.0)) / STANDARD_GRAVITY
        return torch.where(any_observed, ref, torch.zeros_like(ref)), any_observed

    def _localization_reference(self, batch: UniversalSensorBatch, query):
        """Planar endpoint displacement from measured speed and yaw rate.

        The previous reference ``sum(v dt)`` is travelled path length, not
        endpoint displacement on curved trajectories.  Here each speed patch
        supplies an arc length ``ds`` and each aligned yaw-rate patch supplies
        ``dpsi``.  Constant-curvature arc integration is exact within a patch
        for constant speed/rate and reduces to straight-line integration as
        dpsi -> 0.  If yaw rate is unavailable, the physically conservative
        straight-line/path-length fallback is retained.
        """
        if canonical_name(query.quantity) != "displacement" or batch.integral_values is None:
            return self._zero(batch, query)

        vel_id = ontology_id("velocity", QUANTITY_TO_ID)
        scalar_id = ontology_id("scalar", AXIS_TO_ID)
        body_id = ontology_id("vehicle_body", LOCATION_TO_ID)
        yaw_rate_id = ontology_id("angular_rate", QUANTITY_TO_ID)
        yaw_axis_id = ontology_id("yaw", AXIS_TO_ID)

        speed_match = (
            batch.token_mask.bool()
            & (batch.channel_ids >= 0)
            & (batch.quantity_ids == vel_id)
            & (batch.axis_ids == scalar_id)
            & (batch.location_ids == body_id)
        )
        yaw_match = (
            batch.token_mask.bool()
            & (batch.channel_ids >= 0)
            & (batch.quantity_ids == yaw_rate_id)
            & (batch.axis_ids == yaw_axis_id)
            & (batch.location_ids == body_id)
        )

        b = batch.features.shape[0]
        ref = batch.features.new_zeros((b,))
        ok = torch.zeros((b,), dtype=torch.bool, device=batch.features.device)

        for row in range(b):
            sm = speed_match[row]
            if not torch.any(sm):
                continue

            # All tokenizer patches share a physical-time grid. Aggregate
            # equivalent/redundant speed sensors at each patch time by mean so
            # sensor duplication cannot multiply the travelled distance.
            times = torch.unique(batch.times_sec[row][sm], sorted=True)
            x = batch.features.new_tensor(0.0)
            y = batch.features.new_tensor(0.0)
            heading = batch.features.new_tensor(0.0)

            for t in times:
                at_t_speed = sm & torch.isclose(
                    batch.times_sec[row], t, rtol=0.0, atol=1e-7
                )
                ds = batch.integral_values[row][at_t_speed].mean()

                ym = yaw_match[row] & torch.isclose(
                    batch.times_sec[row], t, rtol=0.0, atol=1e-7
                )
                dpsi = (
                    batch.integral_values[row][ym].mean()
                    if torch.any(ym)
                    else batch.features.new_tensor(0.0)
                )

                # Exact local displacement for a constant-curvature arc with
                # arc length ds and heading change dpsi. Use the analytic
                # straight-line limit near zero to avoid numerical cancellation.
                if float(torch.abs(dpsi)) < 1e-6:
                    local_x = ds
                    local_y = ds * 0.0
                else:
                    local_x = ds * torch.sin(dpsi) / dpsi
                    local_y = ds * (1.0 - torch.cos(dpsi)) / dpsi

                c = torch.cos(heading)
                sn = torch.sin(heading)
                x = x + c * local_x - sn * local_y
                y = y + sn * local_x + c * local_y
                heading = heading + dpsi

            ref[row] = torch.sqrt(x.square() + y.square())
            ok[row] = True

        return ref, ok

    def __call__(self, batch: UniversalSensorBatch, queries):
        b = batch.features.shape[0]
        qn = len(queries)
        reference = batch.features.new_zeros((b, qn))
        observed = torch.zeros((b, qn), dtype=torch.bool, device=batch.features.device)
        for j, query in enumerate(queries):
            rule = self._rules.get(canonical_name(query.target_type), self._zero)
            value, mask = rule(batch, query)
            reference[:, j] = value
            observed[:, j] = mask
        return reference, observed
