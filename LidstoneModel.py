# from utils import create_counter_dict
#
#
# class LidstoneModel:
#     def __init__(self, words_list, X, split_factor=0.9):
#         if split_factor > 1 or split_factor < 0:
#             raise Exception('Invalid split factor')
#         self.__X = X
#         self.__total_size = len(words_list)
#         self.__training_size = round(self.__total_size * split_factor)
#         self.__validation_size = self.__total_size - self.__training_size
#         self.__training = create_counter_dict(words_list[:self.__training_size])
#         self.__validation = create_counter_dict(words_list[self.__training_size:])
#         self.__create_prob_dict()
#
#     def get_training_size(self):
#         return self.__training_size
#
#     def get_validation_size(self):
#         return self.__validation_size
#
#     def __create_prob_dict(self):
#         self.prob_dict = dict()
#         for r in range(0, max(self.__training.values()) + 1):
#
#             self.prob_dict[r] = tr / (words_dict.get(event, 0) + lamb) / ( + lamb * X)
#
#     def get_prob(self, event):
#         return self.prob_dict[self.__training.get(event, 0)]
#
#     def __calc_prob(self, r):
#         self.prob_dict[r] = tr / (words_dict.get(event, 0) + lamb) / (sum(words_dict.values()) + lamb * X)
#
#     def debug(self):
#         sum_p = self.get_prob('unseen-word') * self.__N0
#         for event in self.__training.keys():
#             sum_p += self.get_prob(event)
#         return sum_p
