import gym

ENVIRONMENT_SPECS = (
    {
        'id': 'MobileRobotNav-v0',
        'entry_point': ('diffuser.environments.mobile_robot:MobileRobotEnv'),
    },
)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        return gym_ids
    except:
        print('[ diffuser/environments/registration ] WARNING: not registering diffuser environments')
        return tuple()