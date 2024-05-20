"""Schedular to control training status

Schedular json file:
{
    "current_epoch": xxx,
    "end_epoch": xxx,
    "step": xxx,
    "schedular": [xxx, xxx, ...],
    "skip": [xxx, xxx, ...]
}

"""
import os
import json


class Schedular:

    def __init__(self, schedular_dir: str, current_epoch: int, step: int,
                 start_epoch: int = 0, end_epoch: int = None):
        self._schedular_dir = schedular_dir
        self._current_epoch = current_epoch
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._step = step
        self._schedular_path = os.path.join(self._schedular_dir, "schedular.json")

        if os.path.exists(self._schedular_path):
            with open(self._schedular_path, "r") as f:
                self._schedular_data = json.load(f)
        else:
            self._schedular_data = {"schedular": [], "skip": []}

        # update using init args
        self._schedular_data["current_epoch"] = self._current_epoch
        self._schedular_data["start_epoch"] = self._start_epoch
        self._schedular_data["end_epoch"] = self._end_epoch
        self._schedular_data["step"] = self._step

    def _read(self):
        if not os.path.exists(self._schedular_path):
            with open(self._schedular_path, "w") as f:
                json.dump(self._schedular_data, f, indent=2)
            return self._schedular_data
        else:
            with open(self._schedular_path, "r") as f:
                return json.load(f)

    def _update(self):
        schedular_data = self._read()
        schedular_data["current_epoch"] = self._current_epoch
        schedular_data["end_epoch"] = self._end_epoch
        schedular_data["step"] = self._step
        with open(self._schedular_path, "w") as f:
            json.dump(schedular_data, f, indent=2)

    def step(self):
        self._current_epoch += 1
        self._update()
        return self

    def is_active(self):
        schedular_data = self._read()
        if self._current_epoch in schedular_data["schedular"]:
            active = True
        elif self._end_epoch is not None and self._current_epoch > self._end_epoch:
            active = False
        elif (self._current_epoch != 0 and self._current_epoch % self._step == 0) \
                and self._current_epoch not in schedular_data["skip"]:
            active = True
        else:
            active = False
        return active
