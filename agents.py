from gym_cribbage.envs.cribbage_env import (
    CribbageEnv,
    evaluate_cards,
    Deck,
    Card,
    Stack,
    State
)
import numpy as np
import random as rand
import pandas as pd
import logging
import copy
import datetime


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

    def score_hand(self, env, card1, card2, my_strategy, opp_strategy, player_num):
        # simulates the score of a hand given the dealt hand, the cards to discard,
        # the pegging strategy, and the opponent's strategy
        reward_diff = 0
        state = env.state
        prev_phase = None
        while True:
            if state.phase == 0:
                if state.hand_id == player_num:
                    if card1 is not None:
                        state, reward, done, debug = env.step(card1)
                        card1 = None
                    else:
                        state, reward, done, debug = env.step(card2)
                    reward_diff += reward
                else:
                    state, reward, done, debug = env.step(opp_strategy.play(env))
                    reward_diff -= reward
            elif state.phase == 1:
                if state.hand_id == player_num:
                    state, reward, done, debug = env.step(my_strategy.play(env))
                    reward_diff += reward
                else:
                    state, reward, done, debug = env.step(opp_strategy.play(env))
                    reward_diff -= reward
            else:
                if state.hand_id == player_num:
                    state, reward, done, debug = env.step([])
                    reward_diff += reward
                else:
                    state, reward, done, debug = env.step([])
                    reward_diff -= reward

            # end of a hand
            if state.phase == 0 and prev_phase == 2:
                return reward_diff
            prev_phase = state.phase
            # if game over, return reward
            if done:
                return reward_diff


    def simulate_hand(self, player_num, hand, dealer, n_players):
        # create a totally new environment with the same cards dealt
        new_env = CribbageEnv(n_players=n_players, verbose=False, debug=False)
        new_env.reset(dealer=dealer)

        # reset the deck
        new_env.deck = Deck()
        # reset hands
        new_env.hands = [Stack() for i in range(n_players)]
        # add my sim hand to my hand
        new_env.hands[player_num] = hand
        # remove my sim hand from the deck
        for c in hand:
            new_env.deck = new_env.deck.remove(c)
        # deal cards to other players
        for i in range(n_players):
            for j in range(new_env._cards_per_hand):
                # only deal to other players
                if i != player_num:
                    new_env.hands[i].add_(new_env.deck.deal(player=i))

        # reset the state
        new_env.state = State(
            new_env.hands[dealer],
            dealer,
            dealer,
            0,
            0,
            []
        )

        return new_env

    
    def discard(self, env):
        # check if card to discard already calculated
        if not self.next_discard:
            t1 = datetime.datetime.now()
            scores = np.zeros((6, 6))
            hand = env.state.hand

            # loop through all 2 card combinations to discard
            player_num = 0
            hand = copy.deepcopy(env.hands[0])
            dealer = copy.deepcopy(env.dealer)
            n_players = copy.deepcopy(env.n_players)

            # get new environment
            for i in range(5):
                for j in range(i+1, 6):
                    points = []
                    card1 = hand[i]
                    card2 = hand[j]
                    for t in range(self.trials):
                        new_env = self.simulate_hand(player_num, hand, dealer, n_players)
                        # simulate
                        my_strategy = Greedy()
                        opp_strategy = Greedy()
                        reward_diff = self.score_hand(new_env, card1, card2, my_strategy, opp_strategy, player_num)
                        hand = copy.deepcopy(env.hands[0])
                        points.append(reward_diff)
                    scores[i][j] = np.mean(points)

            # find 2 best cards
            # replace non-scores with large negative number
            scores[scores == 0] = -1000
            discard1, discard2 = np.unravel_index(scores.argmax(), scores.shape)
            self.next_discard = hand[int(discard2)]
            # print(datetime.datetime.now() - t1)
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
