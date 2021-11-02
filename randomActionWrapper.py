import random 
import gym

class RandomActionWrapper(gym.ActionWrapper):
    def action(self, action):
        actions = list(self.actions)
        randomNumber = random.randint(0,9)
        if randomNumber == 3 or randomNumber == 5:
            action = random.choice(actions)
        return action

