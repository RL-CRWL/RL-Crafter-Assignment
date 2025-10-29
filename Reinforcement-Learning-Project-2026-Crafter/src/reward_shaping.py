import gymnasium as gym

class CrafterRewardShaping(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_achievements = None
        
    def reset(self, **kwargs):
        # handles seed and options parameters
        obs, info = self.env.reset(**kwargs)
        self.last_achievements = self._get_achievements(info)
        return obs, info
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_achievements = self._get_achievements(info)
        
        # add shaped rewards for progress
        shaped_reward = self._calculate_shaped_reward(current_achievements)
        total_reward = reward + shaped_reward
        
        self.last_achievements = current_achievements
        return obs, total_reward, done, truncated, info
    
    def _get_achievements(self, info=None):
        if info and 'achievements' in info:
            return info['achievements'].copy()
        return {}
    
    def _calculate_shaped_reward(self, current):
        shaped = 0.0
        last = self.last_achievements or {}
        
        # reward for new achievements
        for achievement, count in current.items():
            last_count = last.get(achievement, 0)
            if count > last_count:
                shaped += self._achievement_value(achievement)
        
        # rewards for tool progression
        tool_progression = [
            'collect_wood', 'make_wood_pickaxe', 'collect_stone', 
            'make_stone_pickaxe', 'collect_iron', 'make_iron_pickaxe'
        ]
        
        for i, tool in enumerate(tool_progression):
            if current.get(tool, 0) > last.get(tool, 0):
                shaped += 0.5 * (i + 1)  # increasing reward for progression
        
        return shaped
    
    def _achievement_value(self, achievement):
        values = {
            'collect_wood': 0.1,
            'collect_stone': 0.2,
            'collect_iron': 0.5,
            'collect_coal': 0.3,
            'collect_diamond': 1.0,
            'make_wood_pickaxe': 0.5,
            'make_stone_pickaxe': 1.0,
            'make_iron_pickaxe': 2.0,
            'defeat_skeleton': 1.0,
            'defeat_zombie': 1.5,
            'place_table': 0.3,
            'place_furnace': 0.4,
            'collect_sapling': 0.05,
            'place_plant': 0.05,
        }
        return values.get(achievement, 0.05)