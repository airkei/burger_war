from gym.envs.registration import register

register(
    id='BurgerWarWorld-v0',
    entry_point='dqn_modules.run_normal:BottiNodeEnv'
)