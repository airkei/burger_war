from gym.envs.registration import register

register(
    id='BurgerWarWorld-v0',
    entry_point='dqn_modules.normal_run:BottiNodeEnv'
)

register(
    id='BurgerWarWorldBattle-v0',
    entry_point='dqn_modules.battle_run:BottiNodeEnv'
)