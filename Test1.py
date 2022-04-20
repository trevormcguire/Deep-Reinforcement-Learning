from trainers.BiModels import BiTrainer
from agents.Fatcat import Agent
from environments.General import Environment
from models.MultiTimeFrame_10_20_60 import Model
from utils.load import DataLoader
from utils.rewards import WeightedMean

if __name__ == "__main__":
    
    MODEL_DIR = "Results/Test1"
    DATA_DIR = "PriceData"

    loader = DataLoader(data_dir=DATA_DIR, sma_periods=[9,20,50,100,200], bb_bands=True)

    NUM_TICKERS = int(input("Enter how many tickers you want to run for:\n"))

    data = loader.load_random(num_tickers=NUM_TICKERS)

    NUM_EPISODES = int(input("Enter how many episodes to run for:\n"))

    env = Environment(data=data, buy_on="open", sell_on="open", period=60, reward_function=WeightedMean())
    buyer = Agent(num_feats=11, num_actions=2, max_replay_mem=3000, target_update_thresh=200, replay_batch_size=64, gamma=0.6)
    seller = Agent(num_feats=11, num_actions=2, max_replay_mem=3000, target_update_thresh=200, replay_batch_size=64, gamma=0.1)

    trainer = BiTrainer(environment=env, buyer=buyer, seller=seller, save_dir=MODEL_DIR)

    trainer.train(num_episodes=NUM_EPISODES)


