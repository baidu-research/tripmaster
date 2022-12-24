"""
enum
"""
from enum import Enum, auto


class AutoNamedEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name