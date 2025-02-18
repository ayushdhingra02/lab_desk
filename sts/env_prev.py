from typing import Dict, Tuple, Union

import numpy as np
import math
import mujoco
# print(mujoco.__version__)
import typing
import torch
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, .50)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 67,
    }

    def __init__(
            self,
            xml_file="./kondo_scene_squat_stand.xml",
            frame_skip=5,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=None,
            **kwargs,
    ):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            render_mode=render_mode,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = 0.1
        self._contact_cost_weight = 0.5
        # self._contact_cost_range =
        self._healthy_reward = 10
        self._terminate_when_unhealthy = 0
        self._healthy_z_range = {0,10}

        self._reset_noise_scale = 0
        self.qpos_storage = []
        self.qvel_storage = []
        self.height_threshold=0.08
        self.tilt_threshold = 45
        self.target_height = 0.28
        self.terminated = False
        self.steps=0

        self.num_obs=obs_size
        self.num_privileged_obs=obs_size
        self.num_obs_history=obs_size
        self.num_actions=22
        self.episode_length_buf=torch.tensor([], dtype=torch.int64)
        self.max_episode_length=1000
        self.num_envs=1
        self.num_train_envs=1
        self.category_names=None
        self.curricula=None
        self.num_steps_per_env=1000




    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = -2.75, 2.75
        self.action_space = Box(low=low, high=high, shape=(22,),dtype=np.float32)
        return self.action_space

    @property
    def healthy_reward(self):
        return self.is_healthy() * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    def get_obs(self):
        # Replace with your logic for returning the observation
        return self._get_obs()
    def _get_obs(self):
        # position = self.data.qpos.flatten()
        # velocity = self.data.qvel.flatten()
        #
        # return np.concatenate(
        #     (
        #         position,
        #         velocity,
        #     )
        # )
        delta_t = self.dt
        current_qpos=self.data.qpos.flatten()
        previous_qpos=self.qpos_storage[-1]
        diff=current_qpos-previous_qpos
        calculated_qvel=np.zeros(28)
        calculated_qvel[6:] = diff[7:]/self.dt

        linear_velocity = (current_qpos[:3] - self.qpos_storage[-1][:3]) / (delta_t + 1e-6)  # Shape (3,)

        # Angular velocity (qvel[3:6])
        current_orientation = current_qpos[3:7]  # Quaternion (w, x, y, z)
        previous_orientation = self.qpos_storage[-1][3:7]  # Quaternion (w, x, y, z)
        angular_velocity = self.quaternion_to_angular_velocity(current_orientation, previous_orientation,
                                                               delta_t)  # Shape (3,)

        # Combine linear and angular velocities
        # qvel = np.concatenate((linear_velocity, angular_velocity))
        calculated_qvel[:3] = linear_velocity
        calculated_qvel[3:6] = angular_velocity
        # print("current_qpos:", current_qpos)
        # print("previous_qpos:", previous_qpos)
        # print("diff:", diff)
        # print("calculated_qvel:", calculated_qvel)
        # print("linear_velocity:", linear_velocity)
        # print("angular_velocity:", angular_velocity)

        self.qvel_storage.append(calculated_qvel.copy())
        self.qpos_storage.append(current_qpos.copy())

        # print("qpos_storage:", self.qpos_storage)
        # print("qvel_storage:", self.qvel_storage)

        custom_observation = np.concatenate([self.qpos_storage[-1], calculated_qvel])
        # print("custom_observation:", custom_observation)
        # , self.qpos_storage[-2], self.qvel_storage[-2]

        return custom_observation

    def _reset_episode_storage(self):
        """Reset storage for qpos and qvel at the beginning of a new episode."""
        # Initialize with reset model data
        self.qpos_storage = [self.data.qpos[:29].copy()]  # Assuming model provides the initial position
        self.qvel_storage = [np.zeros(28)]  # Initialize with zeros for velocity

    def _store_step_data(self, qpos, qvel):
        """Store qpos and qvel for each step during the episode."""
        self.qpos_storage.append(qpos.copy())  # Store a copy of qpos
        self.qvel_storage.append(qvel.copy())  # Store a copy of qvel

    def step(self, action):
        # print (self.steps)
        # if self.steps>100 or self.terminated==True:
        #     self.reset_model()
        self.steps += 1
        xy_position_before = mass_center(self.model, self.data)
        action = self._compute_torques(action)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        # terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            }


        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        if self.steps>=100:
            self.terminated=True
        return observation, reward, self.terminated,False, info

    def _compute_torques(self,action):
        kp = 2.0  # Proportional gain
        kd = 1.0  # Derivative gain

        last_qpos = self.qpos_storage[-1]  # Last step qpos
        last_qvel = self.qvel_storage[-1]  # Last step qvel

        desired_pos=action

        position_error=desired_pos-last_qpos[7:29]
        torque = kp * position_error - kd * last_qvel[6:28]

        torque= np.clip(torque, -10, 10)
        return torque

    def quaternion_to_angular_velocity(self,current_orientation, previous_orientation, delta_t):
        # Convert the current and previous orientations (quaternions) to rotation objects
        r_current = R.from_quat(current_orientation)  # Current quaternion
        r_previous = R.from_quat(previous_orientation)  # Previous quaternion

        # Compute the relative rotation (change in orientation)
        r_relative = r_current * r_previous.inv()

        # Convert the relative rotation to a rotation vector (axis-angle representation)
        angular_velocity_vector = r_relative.as_rotvec() / delta_t  # Shape (3,)

        return angular_velocity_vector

    def reward_func(self, x, x_hat, c):
        return -np.exp(c * ((x_hat - x) ** 2))
    def _get_rew(self, x_velocity: float, action):

        w_i = 1 / 17

        # Terms for the reward function
        phi_base = np.array([0, 0, -1])
        h_base = np.array([0, 0, -1])
        v_base = np.array([0, 0, 0])
        tau = self.data.qfrc_actuator
        tau_hat = np.zeros_like(tau)
        q_i = self.data.qvel
        q_dot_hat = np.zeros_like(q_i)
        weights = [1 / 17, 4 / 17, 4 / 17, 1 / 17, 4 / 17, 1 / 17, 4 / 17, 1 / 17, 4 / 17, 4 / 17]
        normalization = [-1.02, -12.5, -2, -0.031, -0.109, 1, -1.02, -5.556, -16.33, -16.33]
        torso_pose = np.array([0, 0, -1])
        head_height = np.array([0, 0, 0.36])  # Assuming head height should be around 1 when standing
        body_ground_contact = 0 if any(contact.geom1 == 'ground' for contact in self.data.contact) else 1

        # Compute the reward terms
        # base_pose_reward = -w_i * np.linalg.norm(phi_base - self.data.qpos[0:3])
        # base_height_reward = -w_i * np.linalg.norm(h_base - self.data.qpos[2])
        # base_velocity_reward = -w_i * np.linalg.norm(v_base - self.data.qvel[0:3])
        # joint_torque_regularization = -w_i * np.linalg.norm(tau)
        #
        # joint_velocity_regularization = -w_i * np.linalg.norm(q_i)
        #
        # body_ground_contact_reward = -w_i * body_ground_contact
        #
        # upper_torso_pose_reward = -w_i * np.linalg.norm(torso_pose - self.data.qpos[3:6])
        # head_height_reward = -w_i * np.linalg.norm(head_height - self.data.qpos[2])

        left_foot_placement_reward = -w_i * np.linalg.norm(
            self.data.site_xpos[self.model.site('left_foot_site').id] - self.data.qpos[0:3])
        right_foot_placement_reward = -w_i * np.linalg.norm(
            self.data.site_xpos[self.model.site('right_foot_site').id] - self.data.qpos[0:3])

        base_pose_reward = weights[0] * np.sum(self.reward_func(self.data.qpos[0:3], phi_base, normalization[0]))
        base_height_reward = weights[1] * np.sum(self.reward_func(self.data.qpos[2], h_base, normalization[1]))
        base_velocity_reward = weights[2] * np.sum(self.reward_func(self.data.qvel[0:3], v_base, normalization[2]))
        joint_torque_regularization = weights[3] * np.sum(self.reward_func(tau, tau_hat, normalization[3]))
        joint_velocity_regularization = weights[4] * np.sum(self.reward_func(q_i, q_dot_hat, normalization[4]))
        body_ground_contact_reward = weights[5] * body_ground_contact
        upper_torso_pose_reward = weights[6] * np.sum(
            self.reward_func(self.data.qpos[3:6], torso_pose, normalization[6]))
        head_height_reward = weights[7] * np.sum(self.reward_func(self.data.qpos[2], head_height, normalization[7]))

        # Sum of all rewards
        reward = (0.2 * base_pose_reward + 0.2 * base_height_reward + 0.2 * base_velocity_reward +
                  0.05 * joint_torque_regularization + 0.05 * joint_velocity_regularization +
                  0.2 * body_ground_contact_reward + 0.05 * upper_torso_pose_reward + 0.2 -
                  0.2 * head_height_reward + 0.001 * left_foot_placement_reward +
                  0.001 * right_foot_placement_reward)

        return reward, []

    def debug_contacts(self,left_foot_contact,right_foot_contact):
        print("Left foot contact: ", self.model.body("LeftFoot").id)
        print("Right foot contact: ", self.model.body("RightFoot").id)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            print(f"Contact {i}: Geom1 = {contact.geom1}, Geom2 = {contact.geom2}")

    def is_body_in_contact(self,body_id):
        for contact in self.data.contact[:self.data.ncon]:
            # Get the geom IDs involved in the contact
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if either geom corresponds to the body
            if self.model.geom_bodyid[geom1]== body_id or self.model.geom_bodyid[geom2] == body_id:
                return True  # Body is in contact

        return False  # No contact found for this body


    def has_fallen(self):
        torso_height = self.data.xpos[self.model.body("Torso").id][2]

        # Get torso orientation (quaternion)
        torso_orientation = self.data.xquat[self.model.body("Torso").id]

        # Calculate tilt angle from upright position
        # Assuming the upright orientation quaternion is approximately [1, 0, 0, 0]
        upright_orientation = np.array([1, 0, 0, 0])
        tilt_angle = np.arccos(np.clip(np.dot(torso_orientation, upright_orientation), -1.0, 1.0)) * (180 / np.pi)

        # Check conditions
        has_fallen_due_to_height = torso_height < self.height_threshold
        has_fallen_due_to_tilt = tilt_angle > self.tilt_threshold
        if has_fallen_due_to_height or has_fallen_due_to_tilt:
            self.terminated=True
        return has_fallen_due_to_height or has_fallen_due_to_tilt

    def calculate_zmp(self):
        m_total = 0
        num_links = self.model.nbody
        zmp_x, zmp_y, zmp_numerator_x, zmp_numerator_y = 0, 0, 0, 0

        for i in range(num_links):
            mass = self.model.body_mass[i]
            pos = self.data.xpos[i]
            acc = (self.data.cacc[i] + self.model.opt.gravity[2])  # acceleration including gravity

            zmp_numerator_x += mass * (pos[0] * acc[2] - pos[2] * acc[0])
            zmp_numerator_y += mass * (pos[1] * acc[2] - pos[2] * acc[1])
            m_total += mass * acc[2]

        zmp_x = zmp_numerator_x / m_total
        zmp_y = zmp_numerator_y / m_total
        return zmp_x, zmp_y

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.steps=0
        self.terminated=False
        # super().reset()
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qpos_squat = [
            0.00326883, 0, 0.215544,
            0.999472, 0, 0.0324943, 0,
            0.00142608, 0.0219053, - 1.31238, 2.34488, - 1.09752, - 0.0219516,
            - 0.00142608, - 0.0219053, - 1.31742, 2.35333, - 1.10093, 0.0219516,
            0, 0,
            - 0.63612, - 0.04712, - 0.37696, - 1.67276,
            - 0.7068, - 0.04712, 0.98952, - 1.46072  # Right leg (hip, knee, ankle)
        ]
        qpos_squat = np.array(qpos_squat)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos_squat, qvel)
        # self.data.reset()
        # self.model.step()  # Update physics calculations
        # self.do_simulation(np.zeros(22),5)
        self._reset_episode_storage()
        observation = np.concatenate((qpos_squat, qvel))
        # left_foot_initial_z = self.data.body("LeftFoot").xpos[2]  # xpos[2] represents the Z-coordinate
        # right_foot_initial_z = self.data.body("RightFoot").xpos[2]  # xpos[2] represents the Z-coordinate
        # print(left_foot_initial_z, right_foot_initial_z)

        return self.get_observations()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def get_observations(self):
        observations = np.concatenate((self.qpos_storage[-1],self.qvel_storage[-1])) # This calls get_obs() which returns the concatenated result
        # print("Returned Observations:", observations)  # Print returned value for debugging
        return observations
    def start_recording(self):
        pass
    def start_recording_eval(self):
        pass
    def pause_recording(self):
        pass
    def pause_recording_eval(self):
        pass
    def get_complete_frames(self):
        pass

    def get_complete_frames_eval(self):
        pass
    # def reset(self, *, seed: typing.Optional[int] = None, options: typing.Optional[dict] = None):
    #     return super().reset()
        # super().set_state()