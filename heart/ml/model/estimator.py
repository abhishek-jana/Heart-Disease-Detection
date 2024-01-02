# import sys

# from pandas import DataFrame
# from sklearn.pipeline import Pipeline

from heart.exception import HeartException
from heart.logger import logging


# class TargetValueMapping:
#     def __init__(self):
#         self.negative: int = 0

#         self.positive: int = 1

#     def to_dict(self):
#         return self.__dict__

#     def reverse_mapping(self):
#         mapping_response = self.to_dict()

#         return dict(zip(mapping_response.values(), mapping_response.keys()))

class TargetValueMapping:
    def __init__(self):
        self.mapping = {"positive": 1, "negative": 0}

    def map_value(self, value):
        return self.mapping.get(value, value)

    def reverse_map(self, value):
        for key, mapped_value in self.mapping.items():
            if mapped_value == value:
                return key
        return value

    def to_dict_mapping(self):
        return self.mapping