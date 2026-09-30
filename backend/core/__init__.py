"""
VISIONSTRA core: detection helpers and intelligent hazard prioritization.

Hazard prioritization modules (all pure Python, no ML or web imports):

    hazard_config  every tunable weight, threshold and timing
    fields         reads both detection dict shapes used in the repo
    risk           stateless scoring, LOW/MEDIUM/HIGH, ranking
    tracking       stable track_id per object across frames
    motion         smoothed distance, approaching/receding, time-to-contact
    pipeline       HazardPrioritizer: one stateful instance per camera stream
    alerts         AlertPolicy: which hazard to speak, and when

See docs/HAZARD_SCORING.md for the methodology.
"""
