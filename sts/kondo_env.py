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

from scipy.spatial import ConvexHull


# com calculation
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
        "render_fps": 1000,
    }

    def __init__(
            self,
            xml_file="./kondo_scene_sts.xml",
            frame_skip=1,
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
        obs_size = obs_size

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.target_height=0.36542
        self.prev_height=None

        self.lfp=None
        self.rfp=None

        self.decimations=8

        self._reset_noise_scale = 0

        self.default_pos = self.model.key_qpos[1].copy()

        self.qpos_storage = []
        self.qvel_storage = []

        self.num_obs=obs_size
        self.num_privileged_obs=11
        self.num_actions=22
        self.num_obs += self.num_actions+3
        self.episode_length_buf = torch.tensor([], dtype=torch.int64)
        self.max_episode_length = 1000
        self.num_envs = 1
        self.num_train_envs = 1
        self.category_names = None
        self.curricula = None
        self.num_steps_per_env = 1000
        self.obs_history_length=10
        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history=np.zeros(self.num_obs_history)


    def step(self, action):

        self.steps += 1
        # xy_position_before = mass_center(self.model, self.data)
        self.action=action
        action=action*1.5
        self.cumulative_torque = 0.0

        self.previous_qpos = self.data.qpos.copy()
        self.previous_qvel = self.data.qvel.copy()
        self.previous_qacc= self.data.qacc.copy()

        for _ in range(self.decimations):
            torque = self._compute_torques(action)
            self.do_simulation(torque, self.frame_skip)
            self.cumulative_torque += np.sum(np.square(torque))  # Accumulate squared torques

        self.current_qpos=self.data.qpos.copy()
        self.current_qvel=self.data.qvel.copy()
        self.current_qacc=self.data.qacc.copy()
        # xy_position_after = mass_center(self.model, self.data)

        # xy_velocity = (xy_position_after - xy_position_before) / self.dt
        # x_velocity, y_velocity = xy_velocity

        self._compute_obs()
        reward, reward_info = self.reward()

        # terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "reward_info": reward_info,
            }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        if self.steps>=200:
            self.terminated=True

        self.prev_action=self.action



        return self.get_observations(), reward, self.terminated,False, info

    def get_observations(self):
        return {"obs" : self.obs, "privileged_obs": self.privileged_obs, "obs_history":self.obs_history}

    def get_geom_name(self, geom_id):
        name_start = self.model.name_geomadr[geom_id]
        return self.model.names[name_start:].split(b'\x00', 1)[0].decode('utf-8')

    def _compute_obs(self):

        delta_t = self.dt * self.decimations

        # self.current_qvel = self.data.qvel.copy()
        # self.current_qacc = self.data.qacc.copy()
        diff=self.current_qpos-self.previous_qpos
        calculated_qvel=np.zeros(28)

        # velocities of joints
        calculated_qvel[6:] = diff[7:] / delta_t

        # base linear velocity
        linear_velocity = (self.current_qpos[:3] - self.qpos_storage[-1][:3]) / (delta_t + 1e-6)  # Shape (3,)

        current_orientation = self.current_qpos[3:7]  # Quaternion (w, x, y, z)
        previous_orientation = self.qpos_storage[-1][3:7]  # Quaternion (w, x, y, z)

        # base angular velocity
        angular_velocity = self._quaternion_to_angular_velocity(current_orientation, previous_orientation,
                                                               delta_t)  # Shape (3,)

        # Combine linear and angular velocities
        calculated_qvel[:3] = linear_velocity
        calculated_qvel[3:6] = angular_velocity

        self.qvel_storage.append(calculated_qvel.copy())
        self.qpos_storage.append(self.current_qpos.copy())

        obs=np.concatenate([self.get_projected_gravity(),self.current_qpos-self.default_pos, calculated_qvel,self.action])

        ###--------- privileged observation

        gravity_vector = self.model.opt.gravity / np.linalg.norm(self.model.opt.gravity)

        # Access floor geom (assuming the first geom is the floor)
        floor_geom_id = self.model.geom("floor").id
        # print(f"floor id: {floor_geom_id}")
        chair_geom_id = self.model.geom("chair").id
        # print(f"chair id :{chair_geom_id}")

        # geom_14_name = self.get_geom_name(14)
        # geom_26_name = self.get_geom_name(26)

        # print("Waist:", self.model.body(self.model.geom(3).bodyid).name)
        # print("Geom 26:", geom_26_name)

        # Get friction values (array of [sliding, torsional, rolling])
        frictionF = self.model.geom_friction[floor_geom_id]
        frictionC = self.model.geom_friction[chair_geom_id]

        # Get elasticity (MuJoCo calls it "elasticity" in geom data)
        elasticityF = self.model.geom_solref[floor_geom_id][0]  # Usually the first solref parameter
        elasticityC = self.model.geom_solref[chair_geom_id][0]

        privileged_obs = np.hstack([gravity_vector, frictionF, elasticityF, frictionC, elasticityC])
        self.obs=obs
        self.privileged_obs=privileged_obs

        self.obs_history = np.concatenate((self.obs_history[self.num_obs:], obs), axis=-1)

        return {"obs":obs, "privileged_obs":privileged_obs}

    def reward(self):

        ### ------------ height reward and height gain reward
        height_weight=10
        height_gain_weight=5
        head_body_id = self.model.body("Head").id
        head_height = self.data.xipos[head_body_id][2]  # Current head height (z-coordinate)

        # Reward for getting closer to the target height
        height_reward = np.exp(-height_weight * abs(head_height - self.target_height))

        # Reward for increasing height compared to previous step
        if self.prev_height is None:
            height_gain_reward = 0  # No change in first step
        else:
            height_gain_reward = height_gain_weight * max(0, head_height - self.prev_height)

        # Update previous height for next step
        self.prev_height = head_height

        # ------------ feet movement and contact broken reward
        contact_reward_weight = 1.0
        sliding_penalty_weight = 2.0
        lifting_penalty_weight = 5.0  # Higher penalty for lifting

        # Get foot body IDs
        right_foot_id = self.model.body("RightFoot").id
        left_foot_id = self.model.body("LeftFoot").id
        # print(f"Right foot id: {right_foot_id}")
        # print(f"Left food id: {left_foot_id}")

        # Current foot positions
        right_foot_pos = self.data.xipos[right_foot_id]
        left_foot_pos = self.data.xipos[left_foot_id]

        # Reward for maintaining contact
        right_contact = any(
            contact.geom1 == self.model.geom("floor").id and self.model.geom(contact.geom2).bodyid == right_foot_id
            for contact in self.data.contact
        )
        # print (f"Right contact: {right_contact}")

        left_contact = any(
            contact.geom1 == self.model.geom("floor").id and self.model.geom(contact.geom2).bodyid == left_foot_id
            for contact in self.data.contact
        )
        # print(f"Left contact: {left_contact}")

        feet_contact_reward = contact_reward_weight * (int(right_contact) + int(left_contact))

        # Penalize unnecessary horizontal movement (x, y) - sliding
        right_foot_sliding = np.linalg.norm(right_foot_pos[:2] - self.rfp[:2])
        left_foot_sliding = np.linalg.norm(left_foot_pos[:2] - self.lfp[:2])

        # Penalize vertical lifting (z)
        right_foot_lifting = max(0, right_foot_pos[2] - self.rfp[2])  # Lift from initial height
        left_foot_lifting = max(0, left_foot_pos[2] - self.lfp[2])

        # Total penalty for foot movement (sliding + lifting)
        feet_movement_penalty = (
                sliding_penalty_weight * (right_foot_sliding + left_foot_sliding) +
                lifting_penalty_weight * (right_foot_lifting + left_foot_lifting)
        )

        # ------------ closeness orientation reward
        orientation_weight=2
        qpos_target= self.model.key_qpos[1].copy()
        qpos_current = self.data.qpos.copy()
        oreintation_reward= orientation_weight *np.exp(-20 * np.linalg.norm(qpos_current - qpos_target) ** 2)
        done_reward=0
        if np.linalg.norm(qpos_current - qpos_target) < 0.7:  # If very close to final position
            done_reward = 5

        # ------------ zmp reward
        zmp_weight=1
        zmp_reward= zmp_weight* self._zmp_reward(self._calculate_zmp(), self._get_support_polygon())

        # ------------ joint limit penalty
        joint_limit_weight = 2.0  # Tune this weight as needed
        joint_limit_penalty = 0

        for j in range(self.model.njnt):
            joint_angle = self.data.qpos[j]
            joint_range = self.model.jnt_range[j]

            # Check if the joint angle exceeds the limits
            if joint_angle < joint_range[0] or joint_angle > joint_range[1]:
                excess = min(abs(joint_angle - joint_range[0]), abs(joint_angle - joint_range[1]))
                joint_limit_penalty += joint_limit_weight * excess ** 2
        joint_limit_penalty*=0

        # ------------ angular momentum penalty
        angular_momentum_weight = 0.1  # Tune this weight as needed
        angular_momentum = np.linalg.norm(self.data.cvel[0])  # Root link angular velocity
        angular_momentum_penalty = angular_momentum_weight * angular_momentum ** 2

        # ------------ time penalty
        time_penalty_weight = 0.01  # Tune this weight as needed
        time_penalty = time_penalty_weight * self.steps

        # ------------ smoothness and velocity penalty
        velocity_penalty_weight = 1.0
        acceleration_penalty_weight = 0.5
        smoothness_penalty_weight = 2.0

        # Get current joint velocities and accelerations
        joint_velocities = self.data.qvel
        joint_accelerations = self.data.qacc

        # Penalize high joint velocities (to reduce fast movements)
        velocity_penalty = velocity_penalty_weight * np.sum(np.square(joint_velocities))

        # Penalize high joint accelerations (to reduce jerky movements)
        acceleration_penalty = acceleration_penalty_weight * np.sum(np.square(joint_accelerations))

        # Penalize abrupt joint angle changes (smoothness penalty)
        if self.previous_qpos is not None:
            joint_angle_diff = self.current_qpos - self.previous_qpos  # Difference from the previous step
            smoothness_penalty = smoothness_penalty_weight * np.sum(np.square(joint_angle_diff))
        else:
            smoothness_penalty = 0  # No penalty on the first step

        # Update the previous joint angles
        self.prev_joint_angles = joint_velocities.copy()

        # Total penalty for jerky or fast movements
        jerkiness_penalty = (velocity_penalty + acceleration_penalty + smoothness_penalty)*0

        ### ------------ torque penalty
        torque_penalty_weight = 0.1
        torque_penalty = torque_penalty_weight * self.cumulative_torque/self.decimations

        total_reward= (height_reward + height_gain_reward + feet_contact_reward + feet_movement_penalty + oreintation_reward + done_reward + zmp_reward - joint_limit_penalty - angular_momentum_penalty - time_penalty - jerkiness_penalty)
        reward_info= {"height_reward": height_reward, "height_gain_reward": height_gain_reward, "feet_contact_reward": feet_contact_reward, "feet_movement_penalty": feet_movement_penalty, "oreintation_reward": oreintation_reward, "done_reward": done_reward, "zmp_reward": zmp_reward, "joint_limit_penalty": joint_limit_penalty, "angular_momentum_penalty": angular_momentum_penalty, "time_penalty": time_penalty, "jerkiness_penalty": jerkiness_penalty}

        return total_reward ,reward_info

        # ### ------------ height reward and height gain reward
        # height_weight = 10
        # height_gain_weight = 5
        # head_body_id = self.model.body("Head").id
        # head_height = self.data.xipos[head_body_id][2]  # Current head height (z-coordinate)
        #
        # # Reward for getting closer to the target height (clipped exponential)
        # height_reward = np.exp(-height_weight * min(1.0, abs(head_height - self.target_height)))
        #
        # # Reward for increasing height compared to previous step
        # if self.prev_height is None:
        #     height_gain_reward = 0  # No change in first step
        # else:
        #     height_gain_reward = height_gain_weight * max(0, head_height - self.prev_height)
        #
        # # Update previous height for next step
        # self.prev_height = head_height
        #
        # ### ------------ feet movement and contact broken reward
        # contact_reward_weight = 1.0
        # sliding_penalty_weight = 2.0
        # lifting_penalty_weight = 5.0
        #
        # right_foot_id = self.model.body("RightFoot").id
        # left_foot_id = self.model.body("LeftFoot").id
        #
        # right_foot_pos = self.data.xipos[right_foot_id]
        # left_foot_pos = self.data.xipos[left_foot_id]
        #
        # # Reward for maintaining contact
        # right_contact = any(
        #     contact.geom1 == self.model.geom("floor").id and self.model.geom(contact.geom2).bodyid == right_foot_id
        #     for contact in self.data.contact
        # )
        # left_contact = any(
        #     contact.geom1 == self.model.geom("floor").id and self.model.geom(contact.geom2).bodyid == left_foot_id
        #     for contact in self.data.contact
        # )
        # feet_contact_reward = contact_reward_weight * (int(right_contact) + int(left_contact))
        #
        # # Penalize unnecessary horizontal movement (x, y) - sliding (clipped to avoid explosion)
        # right_foot_sliding = np.clip(np.linalg.norm(right_foot_pos[:2] - self.rfp[:2]), 0, 1.0)
        # left_foot_sliding = np.clip(np.linalg.norm(left_foot_pos[:2] - self.lfp[:2]), 0, 1.0)
        #
        # # Penalize vertical lifting (z)
        # right_foot_lifting = np.clip(max(0, right_foot_pos[2] - self.rfp[2]), 0, 0.5)
        # left_foot_lifting = np.clip(max(0, left_foot_pos[2] - self.lfp[2]), 0, 0.5)
        #
        # feet_movement_penalty = (
        #         sliding_penalty_weight * (right_foot_sliding + left_foot_sliding) +
        #         lifting_penalty_weight * (right_foot_lifting + left_foot_lifting)
        # )
        #
        # ### ------------ closeness orientation reward
        # orientation_weight = 2
        # qpos_target = self.model.key_qpos[1].copy()
        # qpos_current = self.data.qpos.copy()
        #
        # # Normalized orientation reward (bounded)
        # orientation_reward = orientation_weight / (1 + 10 * np.linalg.norm(qpos_current - qpos_target))
        #
        # done_reward = 5 if np.linalg.norm(qpos_current - qpos_target) < 0.7 else 0
        #
        # ### ------------ zmp reward
        # zmp_weight = 1
        # zmp_reward = zmp_weight * self._zmp_reward(self._calculate_zmp(), self._get_support_polygon())
        #
        # ### ------------ joint limit penalty
        # joint_limit_weight = 2.0
        # joint_limit_penalty = 0
        #
        # for j in range(self.model.njnt):
        #     joint_angle = self.data.qpos[j]
        #     joint_range = self.model.jnt_range[j]
        #     if joint_angle < joint_range[0] or joint_angle > joint_range[1]:
        #         excess = min(abs(joint_angle - joint_range[0]), abs(joint_angle - joint_range[1]))
        #         joint_limit_penalty += joint_limit_weight * min(1.0, excess ** 2)  # Capped penalty
        #
        # ### ------------ angular momentum penalty
        # angular_momentum_weight = 0.1
        # angular_momentum = np.linalg.norm(self.data.cvel[0])
        # angular_momentum_penalty = angular_momentum_weight * min(5.0, angular_momentum ** 2)  # Capped penalty
        #
        # ### ------------ time penalty
        # time_penalty_weight = 0.01
        # time_penalty = time_penalty_weight * self.steps
        #
        # ### ------------ smoothness and velocity penalty
        # velocity_penalty_weight = 1.0
        # acceleration_penalty_weight = 0.5
        # smoothness_penalty_weight = 2.0
        #
        # joint_velocities = self.data.qvel
        # joint_accelerations = self.data.qacc
        #
        # velocity_penalty = velocity_penalty_weight * np.clip(np.sum(np.square(joint_velocities)), 0, 50)
        # acceleration_penalty = acceleration_penalty_weight * np.clip(np.sum(np.square(joint_accelerations)), 0, 20)
        #
        # if self.previous_qpos is not None:
        #     joint_angle_diff = self.current_qpos - self.previous_qpos
        #     smoothness_penalty = smoothness_penalty_weight * np.clip(np.sum(np.square(joint_angle_diff)), 0, 10)
        # else:
        #     smoothness_penalty = 0
        #
        # self.prev_joint_angles = joint_velocities.copy()
        #
        # jerkiness_penalty = velocity_penalty + acceleration_penalty + smoothness_penalty
        #
        # ### ------------ torque penalty
        # torque_penalty_weight = 0.1
        # torque_penalty = torque_penalty_weight * (self.cumulative_torque / (self.decimations + 1e-6))  # Safe division
        #
        # ### ------------ total reward calculation
        # total_reward = (
        #         height_reward + height_gain_reward + feet_contact_reward +
        #         orientation_reward + done_reward + zmp_reward -
        #         (feet_movement_penalty + joint_limit_penalty + angular_momentum_penalty +
        #          time_penalty + jerkiness_penalty + torque_penalty)
        # )
        #
        # # Reward Clipping (to prevent reward explosion)
        # total_reward = np.clip(total_reward, -50, 50)
        #
        # reward_info = {
        #     "height_reward": height_reward, "height_gain_reward": height_gain_reward,
        #     "feet_contact_reward": feet_contact_reward, "feet_movement_penalty": feet_movement_penalty,
        #     "orientation_reward": orientation_reward, "done_reward": done_reward,
        #     "zmp_reward": zmp_reward, "joint_limit_penalty": joint_limit_penalty,
        #     "angular_momentum_penalty": angular_momentum_penalty, "time_penalty": time_penalty,
        #     "jerkiness_penalty": jerkiness_penalty, "torque_penalty": torque_penalty
        # }
        #
        # return total_reward, reward_info

    def get_projected_gravity(self):
        # Gravity vector in world frame
        gravity_world = self.model.opt.gravity / np.linalg.norm(self.model.opt.gravity)  # Normalize for consistency

        # Get the rotation matrix of the base body
        base_id = self.model.body("Torso").id  # Assuming torso is the base
        base_rot_mat = self.data.xmat[base_id].reshape(3, 3)  # Extract 3x3 rotation matrix

        # Project gravity into the base frame
        projected_gravity = base_rot_mat.T @ gravity_world  # Rotate into local frame

        return projected_gravity

    def reset_model(self):

        self.terminated=False
        self.steps=0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nq
        # )
        qpos_squat = self.model.key_qpos[1].copy()
        qpos_squat = np.array(qpos_squat)
        self.current_qpos=qpos_squat
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.current_qvel=qvel
        # self.data.reset()
        self.action=np.zeros(22)
        self.set_state(qpos_squat, qvel)
        # self.model.step()  # Update physics calculations
        # self.do_simulation(np.zeros(22),0)
        right_foot_id = self.model.body("RightFoot").id
        left_foot_id = self.model.body("LeftFoot").id

        # Current foot positions
        self.rfp = self.data.xipos[right_foot_id]
        self.lfp = self.data.xipos[left_foot_id]
        self._reset_episode_storage()
        observation = np.concatenate([qpos_squat, qvel])
        self.obs=np.concatenate([self.get_projected_gravity(),self.current_qpos-self.default_pos, self.current_qvel,self.action])

        ###--------- privileged observation

        gravity_vector = self.model.opt.gravity / np.linalg.norm(self.model.opt.gravity)

        # Access floor geom (assuming the first geom is the floor)
        floor_geom_id = self.model.geom("floor").id
        # print(f"floor id: {floor_geom_id}")
        chair_geom_id = self.model.geom("chair").id
        # print(f"chair id :{chair_geom_id}")

        # geom_14_name = self.get_geom_name(14)
        # geom_26_name = self.get_geom_name(26)

        # print("Waist:", self.model.body(self.model.geom(3).bodyid).name)
        # print("Geom 26:", geom_26_name)

        # Get friction values (array of [sliding, torsional, rolling])
        frictionF = self.model.geom_friction[floor_geom_id]
        frictionC = self.model.geom_friction[chair_geom_id]

        # Get elasticity (MuJoCo calls it "elasticity" in geom data)
        elasticityF = self.model.geom_solref[floor_geom_id][0]  # Usually the first solref parameter
        elasticityC = self.model.geom_solref[chair_geom_id][0]

        privileged_obs = np.hstack([gravity_vector, frictionF, elasticityF, frictionC, elasticityC])
        self.privileged_obs=privileged_obs

        # print(f"obs {len(self.obs)}, pobs: {len(self.privileged_obs)}")
        # left_foot_initial_z = self.data.body("LeftFoot").xpos[2]  # xpos[2] represents the Z-coordinate
        # right_foot_initial_z = self.data.body("RightFoot").xpos[2]  # xpos[2] represents the Z-coordinate
        # print(left_foot_initial_z, right_foot_initial_z)
        self.obs_history = np.concatenate((self.obs_history[self.num_obs:], self.obs), axis=-1)

        return observation

    def _quaternion_to_angular_velocity(self,current_orientation, previous_orientation, delta_t):
        # Convert the current and previous orientations (quaternions) to rotation objects
        r_current = R.from_quat(current_orientation)  # Current quaternion
        r_previous = R.from_quat(previous_orientation)  # Previous quaternion

        # Compute the relative rotation (change in orientation)
        r_relative = r_current * r_previous.inv()

        # Convert the relative rotation to a rotation vector (axis-angle representation)
        angular_velocity_vector = r_relative.as_rotvec() / delta_t  # Shape (3,)

        return angular_velocity_vector

    def _zmp_reward(self, zmp, support_polygon):
        """
        Computes a soft reward to keep the ZMP inside the support polygon.

        Args:
            zmp (np.array): (x, y) position of the ZMP.
            support_polygon (np.array): Array of (x, y) vertices of the support polygon.

        Returns:
            float: Reward value.
        """
        if support_polygon is None or len(support_polygon) < 3:
            return 0  # No valid polygon, no reward

        # Check if ZMP is inside the polygon using a convex hull
        hull = ConvexHull(support_polygon)
        inside = all(np.dot(eq[:-1], zmp) + eq[-1] <= 0 for eq in hull.equations)

        if inside:
            return 1.0  # Full reward if inside

        # Compute minimum distance from ZMP to the support polygon
        distances = np.linalg.norm(support_polygon - zmp, axis=1)
        min_dist = np.min(distances)

        # Soft penalty function (exponential decay)
        return np.exp(-10 * min_dist ** 2)

    def _calculate_zmp(self):
        m_total = 0
        num_links = self.model.nbody
        zmp_x, zmp_y, zmp_numerator_x, zmp_numerator_y = 0, 0, 0, 0

        for i in range(num_links):
            mass = self.model.body_mass[i]
            pos = self.data.xpos[i]



            # acc = (self.data.cacc[i] + self.model.opt.gravity)  # acceleration including gravity
            acc = (self.data.cacc[i])

            zmp_numerator_x += mass * (pos[0] * acc[2] - pos[2] * acc[0])
            zmp_numerator_y += mass * (pos[1] * acc[2] - pos[2] * acc[1])
            m_total += mass * acc[2]

        zmp_x = zmp_numerator_x / m_total
        zmp_y = zmp_numerator_y / m_total
        return zmp_x, zmp_y

    def _get_support_polygon(self):
        foot_names = ["RightFoot", "LeftFoot"]
        contact_points = []

        # Iterate through all contacts
        for contact in self.data.contact:  # Updated to use direct iteration
            geom1, geom2 = contact.geom1, contact.geom2

            # Get body names from geom IDs
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]


            body1_name = self.model.body(body1).name  # Updated to use .body(index).name
            body2_name = self.model.body(body2).name
            # print("body1 :",body1_name)
            # print("body2 :",body2_name)

            # Check if either geom belongs to a foot
            if body1_name in foot_names or body2_name in foot_names:
                contact_points.append(contact.pos[:2])  # Only x, y for support polygon
        # print ("number of points= ",len(contact_points))
        # Convert to numpy array
        if len(contact_points) < 3:
            return None  # Not enough points for a polygon

        contact_points = np.array(contact_points)

        # Compute convex hull
        hull = ConvexHull(contact_points)

        return contact_points[hull.vertices]

    def _compute_torques(self,action):
        kp = 2.0  # Proportional gain
        kd = 1.0  # Derivative gain

        last_qpos = self.qpos_storage[-1]  # Last step qpos
        last_qvel = self.qvel_storage[-1]  # Last step qvel
        action_scaled=action/self.decimations
        desired_pos=action_scaled + self.default_pos[7:29]

        position_error=desired_pos-last_qpos[7:29]
        torque = kp * position_error - kd * last_qvel[6:28]

        torque= np.clip(torque, -10, 10)
        return torque

    def _reset_episode_storage(self):
        """Reset storage for qpos and qvel at the beginning of a new episode."""
        # Initialize with reset model data
        self.qpos_storage = [self.data.qpos[:29].copy()]  # Assuming model provides the initial position
        self.qvel_storage = [np.zeros(28)]  # Initialize with zeros for velocity

    def _store_step_data(self, qpos, qvel):
        """Store qpos and qvel for each step during the episode."""
        self.qpos_storage.append(qpos.copy())  # Store a copy of qpos
        self.qvel_storage.append(qvel.copy())  # Store a copy of qvel

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = -2.35, 2.35
        self.action_space = Box(low=low, high=high, shape=(22,),dtype=np.float32)

