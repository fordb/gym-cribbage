from gym_cribbage.envs.cribbage_env import CribbageEnv, evaluate_cards, Deck, Card
from agents import *
import numpy as np
import random as rand
import pandas as pd
import logging
import copy
import pprint



def play_game(game_id, env, p1, p2):
    # need to manually turn off logging for some reason
    env.logger.setLevel(logging.NOTSET)

    # new game
    state, reward, done, debug = env.reset()
    
    # initialize game logger
    logger = {
        'game_id': [],
        'player1': [],
        'player2': [],
        'dealer': [],
        'p1_deal': [],
        'p2_deal': [],
        'p1_play': [],
        'p2_play': [],
        'p1_show': [],
        'p2_show': [],
        'p1_score': [],
        'p2_score': [],
        'done': []
    }
    
    # initialize hand logger
    hand_logger = {
        'game_id': game_id,
        'player1': p1.name,
        'player2': p2.name,
        'dealer': env.dealer,
        'p1_deal': 0,
        'p2_deal': 0,
        'p1_play': 0,
        'p2_play': 0,
        'p1_show': 0,
        'p2_show': 0,
        'p1_score': 0,
        'p2_score': 0,
        'done': False
    }
    
    prev_phase = None
    while True:
        if state.phase == 0:
            if state.hand_id == 0:
                state, reward, done, debug = env.step(p1.play(env))
                hand_logger['p1_deal'] += reward
            else:
                state, reward, done, debug = env.step(p2.play(env))
                hand_logger['p2_deal'] += reward
        elif state.phase == 1:
            if state.hand_id == 0:
                state, reward, done, debug = env.step(p1.play(env))
                hand_logger['p1_play'] += reward
            else:
                state, reward, done, debug = env.step(p2.play(env))
                hand_logger['p2_play'] += reward
        else:
            if state.hand_id == 0:
                state, reward, done, debug = env.step([])
                hand_logger['p1_show'] += reward
            else:
                state, reward, done, debug = env.step([])
                hand_logger['p2_show'] += reward
        
        # end of a hand
        if state.phase == 0 and prev_phase == 2:
            p1_hand = hand_logger['p1_deal'] + hand_logger['p1_play'] + hand_logger['p1_show']
            p2_hand = hand_logger['p2_deal'] + hand_logger['p2_play'] + hand_logger['p2_show']            
            print("Score Diff: {}".format(p1_hand - p2_hand))
            hand_logger['p1_score'] = env.scores[0]
            hand_logger['p2_score'] = env.scores[1]
            pprint.pprint(hand_logger)
            
            # add to logger
            for key, value in hand_logger.items():
                logger[key].append(value)
            
            # reset hand logger
            hand_logger = {
                'game_id': game_id,
                'player1': p1.name,
                'player2': p2.name,
                'dealer': env.dealer,
                'p1_deal': 0,
                'p2_deal': 0,
                'p1_play': 0,
                'p2_play': 0,
                'p1_show': 0,
                'p2_show': 0,
                'p1_score': 0,
                'p2_score': 0,
                'done': False
            }
        prev_phase = state.phase
        
        if done:
            hand_logger['p1_score'] = env.scores[0]
            hand_logger['p2_score'] = env.scores[1]
            hand_logger['done'] = done
            
            # add to logger
            for key, value in hand_logger.items():
                logger[key].append(value)
            
            return logger


if __name__ == '__main__':
    env = CribbageEnv(verbose=False)
    p1 = MonteCarlo(player_num=0, trials=100, verbose=True)
    p2 = Greedy()

    df = pd.DataFrame()

    # play 1000 games
    for g in range(100):
        print(g)
        logger = play_game(g, env, p1, p2)
        logger_df = pd.DataFrame.from_dict(logger)
        df = df.append(logger_df)
        print(df.tail(1)[['p1_score', 'p2_score']])

    df = df.reset_index()
    df.to_csv("~/Desktop/mc_game-100.csv", index=False)
