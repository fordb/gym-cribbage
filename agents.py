from gym_cribbage.envs.cribbage_env import CribbageEnv, evaluate_cards, Deck, Card
import numpy as np
import random as rand
import pandas as pd
import logging
import copy


class Player(object):
    # random choices
    def __init__(self):
        self.name = 'Random'

    def play(self, env):
        if env.phase == 0:
            return self.discard(env)
        if env.phase == 1:
            return self.peg(env)
    
    def random(self, hand):
        return hand[rand.randint(0, len(hand)-1)]
    
    def discard(self, env):
        hand = self.get_hand(env)
        return self.random(hand)
    
    def peg(self, env):
        hand = self.get_hand(env)
        return self.random(hand)
    
    def get_hand(self, env):
        return env.state.hand


class HighCard(Player):
    # always play highest card
    def __init__(self):
        self.name = 'High Card'
    
    def high_card(self, hand):
        top = None
        top_rank = -1
        for c in hand:
            if c.rank_value > top_rank:
                top = c
                top_rank = c.rank_value
        return top
    
    def discard(self, env):
        hand = self.get_hand(env)
        return self.high_card(hand)

    def peg(self, env):
        hand = self.get_hand(env)
        return self.high_card(hand)


class LowCard(Player):
    # always play lowest card
    def __init__(self):
        self.name = 'Low Card'
    
    def low_card(self, hand):
        bottom = None
        bottom_rank = 10000
        for c in hand:
            if c.rank_value < bottom_rank:
                bottom = c
                bottom_rank = c.rank_value
        return bottom
    
    def discard(self, env):
        hand = self.get_hand(env)
        return self.low_card(hand)

    def peg(self, env):
        hand = self.get_hand(env)
        return self.low_card(hand)


class Greedy(HighCard):
    # play the card that gives the immediate most points
    def __init__(self):
        self.name = 'Greedy'

    def discard(self, env):
        hand = self.get_hand(env)
        # track cards to discard
        top_score = -1

        # if 5 cards left
        if len(hand) == 5:
            for i in range(5):
                _hand = hand
                _hand = _hand.remove(hand[i])
                score = evaluate_cards(_hand)
                if score > top_score:
                    to_discard = hand[i]
                    top_score = score

        # if 6 cards left
        if len(hand) == 6:
            for i in range(5):
                for j in range(i+1, 6):
                    _hand = hand
                    _hand = _hand.remove(hand[i])
                    _hand = _hand.remove(hand[j])
                    score = evaluate_cards(_hand)
                    if score > top_score:
                        to_discard = hand[i]
                        top_score = score
        return to_discard

    def peg(self, env):
        hand = self.get_hand(env)
        # track points
        points = []
        for c in hand:
            _env = copy.deepcopy(env)
            _env.verbose = False
            _env.debug = False
            _, reward, _, _ = _env.step(c)
            points.append(reward)
        # if no points to be had, discard highest card
        if sum(points) == 0:
            return self.high_card(env.state.hand)
        else:
            return hand[points.index(max(points))]


class MonteCarlo(Greedy):
    # simulate remaining hand and play card with the highest expected value
    def __init__(self, player=0, trials=10):
        self.name = 'Monte Carlo'
        self.trials = trials
        self.next_discard = None
        if player == 0:
            self.p2 = Greedy()
        else:
            self.p1 = Greedy()
        self.player = player
    
    def discard(self, env):

        # check if card to discard already calculated
        if not self.next_discard:
            scores = np.zeros((6, 6))
            hand = env.state.hand            
            
            # loop through all 2 card combinations to discard
            for n in range(self.trials):
                for i in range(5):
                    for j in range(i+1, 6):
                        point_diff = 100

                        state = env.state
                        
                        # create copy of env
                        _env = copy.deepcopy(env)
                        _env.verbose = False
                        _env.debug = False

                        # play the rest of the hand
                        card1 = _env.state.hand[i]
                        card2 = _env.state.hand[j]

                        prev_phase = None
                        end_of_hand = False
                        played = False
                        while not end_of_hand:
                            if _env.phase == 0:
                                if state.hand_id == self.player:
                                    if not played:
                                        state, reward, done, debug = _env.step(card1)
                                        played = True
                                    else:
                                        state, reward, done, debug = _env.step(card2)
                                    point_diff += reward
                                else:
                                    # assume opponent plays greedily
                                    state, reward, done, debug = _env.step(self.p2.play(_env))
                                    point_diff -= reward
                            elif _env.phase == 1:
                                if state.hand_id == self.player:
                                    # assume you peg greedily
                                    state, reward, done, debug = _env.step(self.p2.play(_env))
                                    point_diff += reward
                                else:
                                    # assume opponent pegs greedily
                                    state, reward, done, debug = _env.step(self.p2.play(_env))
                                    point_diff -= reward
                            else:
                                if state.hand_id == self.player:
                                    state, reward, done, debug = _env.step([])
                                    point_diff += reward
                                else:
                                    state, reward, done, debug = _env.step([])
                                    point_diff -= reward

                            # check if end of the hand
                            if state.phase == 0 and prev_phase == 2:
                                scores[i][j] += point_diff
                                end_of_hand = True
                        
                            # check if game ended
                            if done:
                                scores[i][j] += point_diff
                                end_of_hand = True
                                
                            prev_phase = state.phase
                                

            # calculate average score
            scores /= self.trials

            # find 2 best cards
            discard1, discard2 = np.unravel_index(scores.argmax(), scores.shape)
            self.next_discard = hand[int(discard2)]
            return hand[int(discard1)]
        else:
            next_discard = self.next_discard
            self.next_discard = None
            return next_discard


class Human(Player):
    # play as a human
    def __init__(self):
        self.name = 'Human'
        
    def print_menu(self, env):
        hand = self.get_hand(env)
        print("GAME\tPlayer {n}'s available cards".format(n=env.state.hand_id))
        print(5 * "-")
        i = 1
        for c in hand:
            print("{num}. {card}".format(num=i, card=c))
            i += 1
        print(5 * "-")

    def choose_card(self, env):
        hand = self.get_hand(env)
        loop = True
        while loop:
            self.print_menu(env)
            try:
                choice = int(input("GAME\tChoose a card to play [1-{n}]: ".format(n=len(hand))))
                if choice >= 1 and choice <= len(hand):
                    loop = False
                else:
                    print("GAME\tNot a valid selection. Try again...\n")
                return hand[choice - 1]
            except ValueError:
                print("GAME\tNot a valid selection. Try again...\n")

    
    def discard(self, env):
        return self.choose_card(env)

    def peg(self, env):
        return self.choose_card(env)
