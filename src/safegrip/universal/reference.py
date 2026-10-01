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
    * localization    -> integrated measured body speed for displacement.

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
        if canonical_name(query.quantity) != "displacement" or batch.integral_values is None:
            return self._zero(batch, query)
        vel_id = ontology_id("velocity", QUANTITY_TO_ID)
        scalar_id = ontology_id("scalar", AXIS_TO_ID)
        body_id = ontology_id("vehicle_body", LOCATION_TO_ID)
        match = (
            batch.token_mask.bool()
            & (batch.channel_ids >= 0)
            & (batch.quantity_ids == vel_id)
            & (batch.axis_ids == scalar_id)
            & (batch.location_ids == body_id)
        )

        # Integrate each matching physical channel independently across its
        # non-overlapping token patches, then average equivalent sensors.  This
        # keeps sensor-order invariance and avoids multiplying displacement when
        # redundant speed channels are present.
        b = batch.features.shape[0]
        ref = batch.features.new_zeros((b,))
        ok = torch.zeros((b,), dtype=torch.bool, device=batch.features.device)
        for row in range(b):
            ids = torch.unique(batch.channel_ids[row][match[row]])
            ids = ids[ids >= 0]
            if len(ids) == 0:
                continue
            per_channel = []
            for channel_id in ids:
                cm = match[row] & (batch.channel_ids[row] == channel_id)
                per_channel.append(batch.integral_values[row][cm].sum())
            ref[row] = torch.stack(per_channel).mean()
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
