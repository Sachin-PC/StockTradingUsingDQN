from enum import IntEnum
from typing import Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
from itertools import product
import numpy as np


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`
    """
    register(id="stocktrading-v0", entry_point="env:StockMarketEnv", max_episode_steps=1000000)


class StockAction(IntEnum):
    """StockAction"""

    BUY = 0
    SELL = 1
    HOLD = 2


def stockAction_map(action: StockAction) -> int:
    """
    Helper function to map StockAction to its correspoinding Value
    Args:
        action (Action): taken action
    Returns:
        dxdy (int): action value
    """
    mapping = {
        StockAction.BUY: 0,
        StockAction.SELL: 1,
        StockAction.HOLD: 2,
    }
    return mapping[action]


class StockMarketEnv(Env):
    def __init__(self,input_data,starting_investment):

        """
        Stocks Considered = 3
        State = List containing Stocks shares, Its closing prices and remaining cash. Total of 7 values
                if there were n stocks, each state will contain (n*2 + 1) values
        Action: 3 actions can be taken - buy, sell, hold
                Each stock held can perform either of the 3 actions. Hence, action is a list of 3 values
                if generalised to n stocks, action is a list of n values.
        Action Space: for 3 stocks with each taking 3 different possible action, the action space is 3^3.
                if generalised to n stocks, it is 3^n
        """

        self.stock_data = input_data

        self.starting_investment = starting_investment
        self.number_of_records, self.unique_stocks = self.stock_data.shape
        self.action_space = np.arange(3**self.unique_stocks, dtype = int)
        actions_combination = list(product([0,1,2], repeat=len(StockAction)))
        self.actions = np.array([list(action_combination) for action_combination in actions_combination])
        self.state_dim = self.unique_stocks * 2 + 1
        self.agent_step = 0
        self.current_shares_held = np.zeros(self.unique_stocks)
        self.shares_price = self.stock_data[self.agent_step]
        self.balance = self.starting_investment
        self.agent_state = self._get_current_state()
        self.protfolio_value = self.starting_investment

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_step = 0
        self.current_shares_held = np.zeros(self.unique_stocks)
        self.shares_price = self.stock_data[self.agent_step]
        self.balance = self.starting_investment
        self.agent_state = self._get_current_state()
        self.protfolio_value = self.starting_investment
        return self.agent_state


    def step(self, action_index):
        """Take one step in the environment.
        """
        self.agent_step += 1
        self.shares_price = self.stock_data[self.agent_step]
        remaining_balance = self.balance
        action = self.actions[action_index]
        stocks_to_sell = np.where(action == stockAction_map(StockAction.SELL))[0]
        stocks_to_buy = np.where(action == stockAction_map(StockAction.BUY))[0]
        for stock in stocks_to_sell:
            shares_held = self.current_shares_held[stock]
            share_price = self.shares_price[stock]
            remaining_balance += shares_held*share_price
            self.current_shares_held[stock] = 0

        flag_count = 0
        stocks_flag = np.zeros(self.unique_stocks)
        while flag_count != len(stocks_to_buy):
            for stock in stocks_to_buy:
                if stocks_flag[stock] == 0:
                    share_price = self.stock_data[self.agent_step][stock]
                    if remaining_balance > share_price:
                        self.current_shares_held[stock] += 1
                        remaining_balance -= share_price
                    else:
                        stocks_flag[stock] = 1
                        flag_count += 1

        self.balance = remaining_balance

        old_portfolio_value = self.protfolio_value
        self.protfolio_value = self.shares_price.dot(self.current_shares_held) + self.balance
        reward = self.protfolio_value - old_portfolio_value

        self.agent_state = self._get_current_state()

        done = False
        if self.agent_step == self.number_of_records -1:
            done = True

        return self.agent_state, reward, done, self.protfolio_value
    

    def _get_current_state(self):
        return np.concatenate((self.current_shares_held, self.shares_price, np.array([self.balance])))
