import gym

def rgba_to_rgb(rgba):
    return np.delete(rgba, 3, axis=2)

HEIGHT = 720
WIDTH = 1280
ACCEL = 0.1 # m/s^2
MAX_VEL = 1.5 # m/s
CONTROL_FREQUENCY = 1  # @param {type:"slider", min:1, max:30, step:1}
FRAME_SKIP = 60  # @param {type:"slider", min:1, max:30, step:1}


class PiCarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, scene_id):
        super(gym.Env, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=())
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_id
        sim_cfg.enable_physics = True
        sim_cfg.allow_sliding = False

        picam = habitat_sim.CameraSensorSpec()
        picam.uuid = "color_sensor"
        picam.sensor_type = habitat_sim.SensorType.COLOR
        # TODO: Noise model?
        picam.resolution = [HEIGHT, WIDTH]
        picam.hfov = 62.2 # https://elinux.org/Rpi_Camera_Module#Technical_Parameters_.28v.2_board.29
        picam.position = [0.0, 0.076, 0.0] # TODO: maybe change y position since it's not in the center?

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [picam]
        agent_cfg.height = 0.1
        agent_cfg.mass = 0.5
        # agent_cfg.linear_acceleration = 1.4
        
        picar = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(picar)
        self.agent = self.sim.initialize_agent(0)
        
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.target_linear_vel = 0. # m/s
    
    def _random_position(self):
        agent_state = self.agent.get_state()
        agent_state.position = sim.pathfinder.get_random_navigable_point()
        orientation = random.random() * math.pi * 2.0
        agent_state.rotation = utils.quat_from_magnum(
            mn.Quaternion.rotation(-mn.Rad(orientation), mn.Vector3(0, 1.0, 0))
        )

        self.agent.set_state(agent_state)
    
    def _turn(self, direction):
        self.vel_control.angular_velocity = np.array([0, float(direction), 0])
    
    def _update_position(self, time_step):
        current_vel = self.vel_control.linear_velocity[2]

        next_step_vel = None
        if self.target_linear_vel > current_vel:
            next_step_vel = min(MAX_VEL, current_vel + ACCEL * time_step)
        else: 
            next_step_vel = max(-1 * MAX_VEL, current_vel - ACCEL * time_step)
        self.vel_control.linear_velocity = np.array([0., 0., next_step_vel])

        # TODO: probably just directly integrate acceleration into state??

        # Integrate the velocity and apply the transform.
        # Note: this can be done at a higher frequency for more accuracy
        agent_state = self.agent.state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )

        # manually integrate the rigid state
        target_rigid_state = self.vel_control.integrate_transform(
            time_step, previous_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
        end_pos = self.sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )

        # set the computed state
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(
            target_rigid_state.rotation
        )
        self.agent.set_state(agent_state)

        # Check if a collision occured
        dist_moved_before_filter = (
            target_rigid_state.translation - previous_rigid_state.translation
        ).dot()
        dist_moved_after_filter = (
            end_pos - previous_rigid_state.translation
        ).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the filter
        # is _less_ than the amount moved before the application of the filter
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

        # run any dynamics simulation
        self.sim.step_physics(time_step)

        return collided

    def step(self, target_vel):
        self.target_linear_vel= float(target_vel)
        collided = self._update_position(1.0 / (FRAME_SKIP * CONTROL_FREQUENCY))
        
        reward = -1 if collided else abs(target_vel)
        observation = rgba_to_rgb(self.sim.get_sensor_observations()['color_sensor']) 
        
        return (observation, reward, False, None) # observation, reward, done, info
    
    def reset(self):
        return rgba_to_rgb(self.sim.reset()['color_sensor'])  # reward, done, info can't be included
    
    def render(self, mode='human'):
        # TODO
        return
        
    def close (self):
        self.sim.close()
        return