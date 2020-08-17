from gym.envs.registration import register

register(
    id='BurgerWarWorld-v0',
    entry_point='qlearn_modules.botti:BottiNodeEnv'
)