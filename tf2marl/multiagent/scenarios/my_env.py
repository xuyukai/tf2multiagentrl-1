import numpy as np
from tf2marl.multiagent.core import World, Agent, Landmark
from tf2marl.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 3
        num_friends = 3  # ?the number of friends
        num_agents = num_adversaries + num_good_agents + num_friends  # ?add num_friends in agents number
        world.n_adversaries = num_adversaries
        # num_landmarks = 2  #?elimate landmark
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.friend = True if i > num_adversaries + num_good_agents - 1 else False  # ?define friend of agent
            if agent.adversary:
                agent.size = 0.075
            elif agent.friend:
                agent.size = 0.05
            else:
                agent.size = 0.1  # ?define size of each kind of agent
            if agent.adversary:
                agent.accel = 3.0
            elif agent.friend:
                agent.accel = 3.0
            else:
                agent.accel = 4.0  # ?define accel of each kind of agent
            # agent.accel = 20.0 if agent.adversary else 25.0
            if agent.adversary:
                agent.max_speed = 2.0
            elif agent.friend:
                agent.max_speed = 2.0
            else:
                agent.max_speed = 2.3  # ?define maximun of speed of each kind of agent
        # add landmarks
        # world.landmarks = [Landmark() for i in range(num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d' % i
        #     landmark.collide = True
        #     landmark.movable = False
        #     landmark.size = 0.2
        #     landmark.boundary = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if agent.adversary:
                agent.color = np.array([0.85, 0.35, 0.35])
            elif agent.friend:
                agent.color = np.array([0.35, 0.35, 0.85])
            else:
                agent.color = np.array([0.35, 0.85, 0.35])  # ?color
            # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            #agent.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if not agent.adversary and not agent.friend:  # 是good_agent
                agent.state.p_pos = np.random.uniform(-0.4, +0.4, world.dim_p)
            elif agent.friend:
                agent.state.p_pos = [np.random.uniform(0.4, 0.6) * np.random.choice((-1, 1)),
                                     np.random.uniform(0.4, 0.6) * np.random.choice((-1, 1))]
            else:
                agent.state.p_pos = [np.random.uniform(0.6, 0.9) * np.random.choice((-1, 1)),
                                     np.random.uniform(0.6, 0.9) * np.random.choice((-1, 1))]
        # for i, landmark in enumerate(world.landmarks):
        #     if not landmark.boundary:
        #         landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
        #         landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # ?return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary and agent.friend]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    # ?return all friendly agents
    def friends(self, world):
        return [agent for agent in world.agents if agent.friend]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        if agent.adversary:
            main_reward = self.adversary_reward(agent, world)
        elif agent.friend:
            main_reward = self.friend_reward(agent, world)
        else:
            main_reward = self.agent_reward(agent, world)  # ?main reward
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        friends = self.friends(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries and friends:
                if self.is_collision(a, agent):
                    rew -= 100

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 1000000 * self.bound(x)
        return rew

    # adversary 是希望撞上good_agent，但是不被friend撞上的
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        friends = self.friends(world)  # ?add friends obs in reward
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
                # ? keep far from friends
                rew += 0.1 * min([np.sqrt(np.sum(np.square(fd.state.p_pos - adv.state.p_pos))) for fd in friends])

        if agent.collide:  # 如果当前agent是adversary的话
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 100
            for adv in adversaries:
                for fd in friends:
                    if self.is_collision(adv, fd):
                        rew -= 10  # ?航天器的悲欢各不相同
            for adv1 in adversaries:
                for adv2 in adversaries:
                    if self.is_collision(adv1, adv2):
                        rew -= 100

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 1000000 * self.bound(x)
        return rew

    # ?friend reward
    def friend_reward(self, agent, world):
        rew = 0
        shape = False
        friends = self.friends(world)
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # ?friend get 10 reward if it collide with adv, get -100 rewards if agent is collided
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for fd in friends:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(fd.state.p_pos - adv.state.p_pos))) for adv in adversaries])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew -= 100
            for adv in adversaries:
                for fd in friends:
                    if self.is_collision(adv, fd):
                        rew += 10
            for ag in agents:
                for fd in friends:
                    if self.is_collision(ag, fd):
                        rew -= 100
            for fd1 in friends:
                for fd in friends:
                    if self.is_collision(fd1, fd):
                        rew -= 100

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 1000000 * self.bound(x)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def bound(self, x):
        if x < 0.5:
            return 0
        if x < 1.0:
            return (x - 0.5) * 10
        return min(np.exp(2 * x - 2), 10)
