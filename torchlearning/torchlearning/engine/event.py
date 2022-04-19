from enum import Enum


class Event(Enum):
    STARTED = 'engine_stated'
    COMPLETED = 'engine_complete'
    EPOCH_STARTED = 'epoch_started'
    EPOCH_COMPLETED = 'epoch_completed'
    STAGE_STARTED = 'stage_started'
    STAGE_COMPLETED = 'stage_completed'
    ITER_STARTED = 'iter_started'
    ITER_COMPLETED = 'iter_completed'
