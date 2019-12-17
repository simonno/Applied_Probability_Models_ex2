import math
from datetime import datetime
from decimal import Decimal

from utils import create_counter_dict


class LidstoneModel:
    def __init__(self, words_list, X, split_factor=0.9):
        if split_factor > 1 or split_factor < 0:
            raise Exception('Invalid split factor')
        self.__X = X
        self.__total_size = len(words_list)
        self.__training_size = round(self.__total_size * split_factor)
        self.__validation_size = self.__total_size - self.__training_size
        self.__training = create_counter_dict(words_list[:self.__training_size])
        self.__validation = create_counter_dict(words_list[self.__training_size:])
        self.__create_prob_dict()

    def get_training_size(self):
        return self.__training_size

    def get_training_different_events(self):
        return len(self.__training.keys())

    def get_event_appearance(self, event):
        return self.__training.get(event, 0)

    def get_validation_size(self):
        return self.__validation_size

    def __create_prob_dict(self):
        self.__prob_dict = dict()
        self.__perplexity_dict = dict()
        stop = Decimal('2.0')
        step = Decimal('0.01')
        curr_lamb = Decimal('0.0')
        while curr_lamb <= stop:
            self.__prob_dict[curr_lamb] = {0: self.__calc_prob(0, curr_lamb)}
            for r in self.__training.values():
                if r in self.__prob_dict[curr_lamb].keys():
                    continue
                self.__prob_dict[curr_lamb][r] = self.__calc_prob(r, curr_lamb)
            self.__perplexity_dict[curr_lamb] = self.__calc_perplexity(curr_lamb)
            curr_lamb += step

    def get_prob(self, event, lamb):
        return self.__prob_dict[lamb][self.__training.get(event, 0)]

    def get_min_perplexity(self, start, stop, step):
        min_perplexity = float(math.inf)
        lamb = 0
        curr_lamb = start
        while curr_lamb <= stop:
            curr_perplexity = self.get_perplexity(curr_lamb)
            if curr_perplexity < min_perplexity:
                min_perplexity = curr_perplexity
                lamb = curr_lamb
            curr_lamb += step

        return min_perplexity, lamb

    def get_perplexity(self, lamb):
        if lamb in self.__perplexity_dict.keys():
            return self.__perplexity_dict[lamb]
        else:
            perplexity = self.__calc_perplexity(lamb)
            self.__perplexity_dict[lamb] = perplexity
            return perplexity

    def __calc_prob(self, r, lamb):
        return (r + lamb) / (self.get_training_size() + lamb * self.__X)

    def __calc_perplexity(self, lamb):
        start_perp = datetime.now()
        sum_p = 0
        for event, count in self.__validation.items():
            prob = self.get_prob(event, lamb)
            if prob == 0:
                sum_p = float(-math.inf)
                break
            else:
                sum_p += count * math.log(prob)
        prep = math.pow(2, -sum_p / self.get_validation_size())
        print('perplexity {0} - lambda {1} running time: {2}'.format(prep, lamb, datetime.now() - start_perp))
        return prep

    def debug(self, lamb):
        sum_p = self.get_prob('unseen-word', lamb) * (self.__X - self.get_training_different_events())
        for event in self.__training.keys():
            sum_p += self.get_prob(event, lamb)
        return sum_p
