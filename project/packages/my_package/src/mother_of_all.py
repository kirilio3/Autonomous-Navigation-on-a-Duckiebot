#!/usr/bin/env python3

from abc import ABC, abstractmethod


class MotherOfAll(ABC):

    @abstractmethod
    def camera_info_setter(self, distortion_coeffs, camera_matrix):
        pass

    class getStopException(Exception):
        def __init__(self, message="Stop"):
            self.message = message
            super().__init__(self.message)