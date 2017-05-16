# Evolve a control/reward estimation network for the OpenAI Gym using NEAT and AC Learning
from __future__ import print_function

import multiprocessing
import os
import pickle
import argparse
import logging
import sys
import gym.wrappers as wrappers
import matplotlib.pyplot as plt
import neat
import numpy as np
import gym
import math

import visualize


score_range = []


class NeatACNetwork(object):
    def __init__(self, nn, dimension):
        self.nn = nn
        self.dimension = dimension
        self.valueFunction = ValueFunction(dimension)
        self.policy = GaussianPolicy(dimension)
        self.fitness = 0
        self.gamma = 0.99
        self.alpha = 0.1
        self.beta = 0.0005

    def get_value_function(self):
        return self.valueFunction

    def get_policy(self):
        return self.policy

    def get_network(self):
        return self.nn

    def get_fitness(self):
        return self.fitness

    def update(self, old_state, taken_action, new_state, reward):
        # calculate TD error. For critic
        old_state_features = self.nn.activate(old_state)
        new_state_features = self.nn.activate(new_state)
        delta = reward + self.gamma * self.valueFunction.get_value(old_state_features) - self.valueFunction.get_value(new_state_features)

        # Update critic parameters
        delta_omega = (self.alpha * delta) * np.array(old_state_features)
        self.valueFunction.update_parameters(delta_omega)

        # Update policy parameters
        dlogpi = self.policy.dlogPi(old_state_features, taken_action)
        delta_theta = (self.beta * delta) * np.array(dlogpi)
        self.policy.update_parameters(delta_theta)
        self.fitness += reward






class GaussianPolicy(object):
    def __init__(self, dimension):
        self.dimension = dimension
        self.parameters = np.zeros(dimension, dtype=float)
        self.sigma = 1.0

    def get_action(self, state_feature):
        '''
        Perform dot product between state feature and policy parameter and return sample from the normal distribution
        :param state_feature: 
        :return: 
        '''
        mu = np.dot(state_feature, self.parameters)
        return np.random.normal(mu, self.sigma)

    def update_parameters(self, delta):
        for i, param in enumerate(self.parameters):
            self.parameters[i] = param + delta[i]

    def dlogPi(self, state_features, action):
        mu = np.dot(state_features, self.parameters)
        deratives = np.zeros(len(state_features))

        component1 = (action - mu) / math.pow(self.sigma, 2)

        for i, state_feature in enumerate(state_features):
            deratives[i] = state_feature + component1

        return deratives



class ValueFunction(object):
    def __init__(self, dimension):
        self.dimension = dimension
        self.parameters = np.zeros(dimension, dtype=float)

    def get_value(self, state_feature):
        return np.dot(self.parameters, state_feature)

    def update_parameters(self, delta):
        for i, param in enumerate(self.parameters):
            self.parameters[i] = param + delta[i]



class NeatAC(object):
    def __init__(self, config):
        pop = neat.Population(config)
        self.stats = neat.StatisticsReporter()
        pop.add_reporter(self.stats)
        pop.add_reporter(neat.StdOutReporter(True))
        # Checkpoint every 10 generations or 900 seconds.
        pop.add_reporter(neat.Checkpointer(10, 900))
        self.config = config
        self.population = pop
        self.pool = multiprocessing.Pool()

    def execute_algorithm(self, generations):
        self.population.run(self.fitness_function, generations)

    def fitness_function(self, genomes, config):
        '''
        This method is called every generation.
        Create new array 
        :param genomes: 
        :param config: 
        :return: 
        '''
        nets = []
        for gid, g in genomes:
            # create network
            network = neat.nn.FeedForwardNetwork.create(g, config)
            neatNetwork = NeatACNetwork(network, 10) # dimension is hardcoded to be 10 for now. TODO: use config file for dimension

            nets.append((g, neatNetwork))
            g.fitness = []

        scores = []
        for genome, net in nets:
            # run episodes
            episode_count = 0
            MAX_EPISODES = 200
            while True:
                state = env.reset()
                terminal_reached = False
                while not terminal_reached:
                    # get state features
                    state_features = net.get_network().activate(state)
                    # get action based on a policy. I'm using random policy for now
                    action = net.get_policy().get_action(state_features)

                    # take action and observe the reward and state
                    next_state, reward, done, info = env.step([action])
                    # update neural network AC parameters
                    net.update(state, action, next_state, reward)
                    state = next_state
                    if done:
                        terminal_reached = True

                episode_count += 1
                if episode_count >= MAX_EPISODES:
                    break

            scores.append(net.get_fitness())

            # assign fitness to genome
            genome.fitness = net.get_fitness()/episode_count

        score_range.append((min(scores), np.mean(scores), max(scores)))

        print(min(map(np.min, score_range)), max(map(np.max, score_range)))


if __name__ == '__main__':

    # 2 inputs and 10 outputs from NN

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCarContinuous-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)
    print("action space: {0!r}".format(env.action_space))
    print("observation space: {0!r}".format(env.observation_space))

    # Limit episode time steps to cut down on training time.
    # 400 steps is more than enough time to land with a winning score.
    print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
    env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = 400
    print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/actor-critic-data'
    env = wrappers.Monitor(env, directory=outdir, force=True)

    # run the algorithm

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    agent = NeatAC(config)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    while 1:
        try:
            # Run for 100 generations
            agent.execute_algorithm(1)

            visualize.plot_stats(agent.stats, ylog=False, view=False, filename="fitness.svg")

            if score_range:
                S = np.array(score_range).T
                plt.plot(S[0], 'r-')
                plt.plot(S[1], 'b-')
                plt.plot(S[2], 'g-')
                plt.grid()
                plt.savefig("NEATAC-score-ranges.svg")
                plt.close()

            mfs = sum(agent.stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))

            mfs = sum(agent.stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0}".format(mfs))

            # Use the five best genomes seen so far as an ensemble-ish control system.
            best_genomes = agent.stats.best_unique_genomes(5)
            best_networks = []
            for g in best_genomes:
                network = neat.nn.FeedForwardNetwork.create(g, config)
                best_networks.append(NeatACNetwork(network, 10))


            print("Solved.")

            # Save the winners.
            for n, g in enumerate(best_genomes):
                name = 'winner-{0}'.format(n)
                with open(name + '.pickle', 'wb') as f:
                    pickle.dump(g, f)

                visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
                visualize.draw_net(config, g, view=False, filename="-net-enabled.gv",
                                   show_disabled=False)
                visualize.draw_net(config, g, view=False, filename="-net-enabled-pruned.gv",
                                   show_disabled=False, prune_unused=True)


        except KeyboardInterrupt:
            print("User break.")
            break

    env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
