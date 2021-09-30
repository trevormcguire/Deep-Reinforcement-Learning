
from copy import deepcopy
import numpy as np

class BiTrainer(object):
    def __init__(self, environment, buyer, seller, save_dir):
        self.env = environment
        self.buyer = buyer
        self.seller = seller
        self.model_dir = save_dir

    def train(self, num_episodes: int):
        times_held = []

        for n in range(1, num_episodes + 1):
            time_held = self.run_episode()
            times_held.append(time_held)
            if n % 1000 == 0 and n > 0:
                self.buyer.save_model(f"{self.model_dir}/bifatcat_buyer_e{n}_p{sum(self.env.episode_profits)*100:.2f}.pt")
                self.seller.save_model(f"{self.model_dir}/bifatcat_seller_e{n}_p{sum(self.env.episode_profits)*100:.2f}.pt")
        #------------
        summary_data = {
            "Num Episodes": num_episodes, 
            "Total profits": sum(self.env.episode_profits)*100,
            "Mean Profit": np.mean(self.env.episode_profits)*100,
            "Max Time Held": np.max(times_held),
            "Min Time Held": np.min(times_held),
            "Average Time Held": np.mean(times_held),
            "Profits": self.env.episode_profits
            }
        for k,v in summary_data.items():
            if k != "Profits":
                print(f"{k}: {v}")

        with open(f"{self.model_dir}/bifatcat_e{num_episodes}_summary.txt", "w") as f:
            f.write(str(summary_data))

        print(f"Saved summary at {self.model_dir}")
        #------------

    def run_episode(self):
        self.env.reset()
        done = False
        buyer_memory, seller_memory = [], []
        seller_memory = []
        bought_state = None
        bought_next_state = None
        time_held = 0
        while not done:
            current_state = self.env.get_state()

            if not self.env.holding_position:
                action = self.buyer.action(current_state)
                reward, next_state, done = self.env.step(action)

                if action == 1:
                    #delay reward until we sell
                    bought_state = deepcopy(current_state)
                    bought_next_state = deepcopy(next_state)
                else:
                    buyer_memory.append((current_state, action, reward, next_state, done))
                
            else:
                action = self.seller.action(current_state)
                if action == 1:
                    action += 1
                reward, next_state, done = self.env.step(action)
                if action == 2:
                    buyer_memory.append((bought_state, 1, reward, bought_next_state, done))
                
                seller_memory.append((current_state, action, reward, next_state, done))
            time_held += 1

        if not next_state is None:
            self.buyer.replay_memory += buyer_memory
            self.seller.replay_memory += seller_memory

            self.buyer.replay()
            self.seller.replay()

            self.buyer.decay_epsilon()
            self.seller.decay_epsilon()

        return time_held
