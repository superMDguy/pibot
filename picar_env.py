import numpy as np
import magnum as mn
import math
import random

import habitat_sim
from habitat_sim.utils import common as utils
import gym

def rgba_to_rgb(rgba):
    return np.delete(rgba, 3, axis=2)

HEIGHT = 368
WIDTH = 640
ACCEL = 1.0 # m/s^2
MAX_VEL = 1.5 # m/s
CONTROL_FREQUENCY = 1 # TODO: implement. Will require separate render modes for video vs. model
FRAME_SKIP = 30


class PiCarEnv(gym.Env):
    def __init__(self, scene_id):
        super(gym.Env, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,)) # np.array([forward, turn])
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
        
        self.reset()
    
    def _random_position(self):
        agent_state = self.agent.get_state()
        agent_state.position = self.sim.pathfinder.get_random_navigable_point()
        orientation = random.random() * math.pi * 2.0
        agent_state.rotation = utils.quat_from_magnum(
            mn.Quaternion.rotation(-mn.Rad(orientation), mn.Vector3(0, 1.0, 0))
        )

        self.agent.set_state(agent_state)
        
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
        # TODO is this necessary??
        self.sim.step_physics(time_step)

        return collided, end_pos

    def step(self, controls):
        if controls is not None: # If controls are None, just update physics
            self.target_linear_vel = float(controls[0])
            self.vel_control.angular_velocity = np.array([0, float(controls[1]), 0])
            
        collided, new_pos = self._update_position(1.0 / (FRAME_SKIP * CONTROL_FREQUENCY))
        reward = -10 if collided else (new_pos - self.start_pos).dot()
        
        observation = self.render()
        
        return (observation, reward, False, dict()) # observation, reward, done, info
    
    def reset(self):
        self.sim.reset()
        
        self._random_position()
        self.target_linear_vel = 0. # m/s
        self.vel_control.linear_velocity = np.zeros((3,))
        self.vel_control.angular_velocity = np.zeros((3,))
        
        self.start_pos = self.agent.get_state().position
        
        return self.render()
    
    def render(self):
        return rgba_to_rgb(self.sim.get_sensor_observations()['color_sensor'])
        
    def close (self):
        return self.sim.close()

    
def build_navmesh(sim):
    navmesh_settings = habitat_sim.NavMeshSettings()

    navmesh_settings.set_defaults()
    navmesh_settings.cell_size = 0.01 #@param {type:"slider", min:0.01, max:0.2, step:0.01}
    #default = 0.05
    navmesh_settings.cell_height = 0.01 #@param {type:"slider", min:0.01, max:0.4, step:0.01}
    #default = 0.2

    #@markdown **Agent parameters**:
    navmesh_settings.agent_height = 0.1 #@param {type:"slider", min:0.01, max:3.0, step:0.01}
    #default = 1.5
    navmesh_settings.agent_radius = 0.1 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
    #default = 0.1
    navmesh_settings.agent_max_climb = 0.2 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
    #default = 0.2
    navmesh_settings.agent_max_slope = 45 #@param {type:"slider", min:0, max:85, step:1.0}
    # default = 45.0
    # fmt: on
    navmesh_settings.filter_low_hanging_obstacles = True  # @param {type:"boolean"}
    # default = True
    navmesh_settings.filter_ledge_spans = True  # @param {type:"boolean"}
    # default = True
    navmesh_settings.filter_walkable_low_height_spans = True  # @param {type:"boolean"}
    # default = True

    navmesh_settings.region_min_size = 0 #@param {type:"slider", min:0, max:50, step:1}
    #default = 20
    navmesh_settings.region_merge_size = 0 #@param {type:"slider", min:0, max:50, step:1}
    #default = 20
    navmesh_settings.edge_max_len = 0 #@param {type:"slider", min:0, max:50, step:1}
    #default = 12.0
    navmesh_settings.edge_max_error = 0.1 #@param {type:"slider", min:0, max:5, step:0.1}
    #default = 1.3
    navmesh_settings.verts_per_poly = 6.0 #@param {type:"slider", min:3, max:6, step:1}
    #default = 6.0
    navmesh_settings.detail_sample_dist = 6.0 #@param {type:"slider", min:0, max:10.0, step:0.1}
    #default = 6.0
    navmesh_settings.detail_sample_max_error = 1.0 #@param {type:"slider", min:0, max:10.0, step:0.1}
    # default = 1.0
    # fmt: on

    navmesh_success = sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=True
    )

    if not navmesh_success:
        print("Failed to build the navmesh! Try different parameters?")