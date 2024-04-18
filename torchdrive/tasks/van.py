from typing import Tuple
from abc import ABC

class Van(ABC):
    def should_log(self, global_step: int, BS: int) -> Tuple[bool, bool]:
        log_text_interval = 1000 // (BS * 20)
        # log_text_interval = 1
        # It's important to scale the less frequent interval off the more
        # frequent one to avoid divisor issues.
        log_img_interval = log_text_interval * 10
        log_img = (global_step % log_img_interval) == 0
        log_text = (global_step % log_text_interval) == 0

        return log_img, log_text
