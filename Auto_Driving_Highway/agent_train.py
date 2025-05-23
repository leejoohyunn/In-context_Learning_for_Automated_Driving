import highway_env
import numpy as np
import gymnasium as gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv  # Import DummyVecEnv
import random
import re
import cv2
import time
from scenario import Scenario
from customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe
)
from analysis_obs import available_action, get_available_lanes, get_involved_cars, extract_lanes_info, extract_lane_and_car_ids, assess_lane_change_safety, check_safety_in_current_lane, format_training_info
import ask_llm
import os

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

class MyHighwayEnv(gym.Env):
    def __init__(self, vehicleCount=15, render_mode = None):
        super(MyHighwayEnv, self).__init__()
        # base setting
        self.vehicleCount = vehicleCount
        self.render_mode = render_mode
        # environment setting
        self.config = {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": vehicleCount,
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(0, 32, 9),
            },
            "duration": 40,
            "vehicles_density": 2,
            "show_trajectories": True,
            "render_agent": True,
            "real_time_rendering" : False,
        }
        
        if render_mode:
            self.config["render_mode"] = render_mode
        self.env = gym.make("highway-v0", render_mode = render_mode)
        self.env.unwrapped.config.update(self.config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(vehicleCount, 5), dtype=np.float32
        # )


    def step(self, action):
        # Step the wrapped environment and capture all returned values
        obs, reward, done, truncated, info = self.env.step(action)
        self.last_observation = obs
        custom_reward = self.calculate_custom_reward(action)
        return obs, custom_reward, done, truncated, info
    
    def render(self, mode = 'rgb_array'):
        """렌더링 함수"""
        return self.env.render()
    
    def set_llm_suggested_action(self, action):
        self.llm_suggested_action = action
    
    def calculate_custom_reward(self, action):
        action_id = action
        action_name = ACTIONS_ALL.get(action_id, "Unknown")
        llm_action = self.llm_suggested_action
    
        # 정확히 일치하는 경우
        if action_name == llm_action:
            reward = 1.0
            print(f"✅ 액션 일치! 보상: {reward}")
            return reward
    
        # 부분적으로 유사한 경우 (속도 관련 액션)
        speed_actions = ["FASTER", "SLOWER"]
        if (action_name in speed_actions and llm_action in speed_actions):
            reward = 0.3
            print(f"⚠️ 액션 부분 일치! 보상: {reward}")
            return reward

        if hasattr(self, '_debug_step') and self._debug_step:
            if action_name == llm_action:
                print(f"✅ 액션 일치! 보상: {reward}")
        
        # 일치하지 않는 경우
        print(f"❌ 액션 불일치! 보상: 0")
        return 0.0
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        # 마지막 관측 저장
        self.last_observation = obs
        # 두 값 모두 돌려줘야 DummyVecEnv가 정상 작동합
        return obs, info
    
    def get_available_actions(self):
        """Get the list of available actions from the underlying Highway environment."""
        sce = Scenario(vehicleCount=self.vehicleCount)
        sce.updateVehicles(self.last_observation, 0)

        toolModels = [
            getAvailableActions(self.env.unwrapped),
            getAvailableLanes(sce),
            getLaneInvolvedCar(sce),
            isChangeLaneConflictWithCar(sce),
            isAccelerationConflictWithCar(sce),
            isKeepSpeedConflictWithCar(sce),
            isDecelerationSafe(sce),
    ]

        available = available_action(toolModels)
        valid_action_ids = [i for i, act in ACTIONS_ALL.items() if available.get(act, False)]
        return valid_action_ids
def save_video_from_frames(frames, filename, fps=10):
    """프레임들을 비디오로 저장"""
    if not frames:
        print("저장할 프레임이 없습니다.")
        return
    
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        # RGB를 BGR로 변환 (OpenCV 형식)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    
    video.release()
    print(f"비디오 저장 완료: {filename}")
    
def main():
    os.makedirs("videos", exist_ok=True)
    env = MyHighwayEnv(vehicleCount=5, render_mode = 'rgb_array')
    observation = env.reset()
    print("Initial Observation:", observation)
    print("Observation space:", env.observation_space)
    # print("Action space:", env.action_space)

    # Wrap the environment in a DummyVecEnv for SB3
    env = DummyVecEnv([lambda: env])  # Add this line
    available_actions = env.envs[0].get_available_actions()
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        exploration_fraction=0.8,  # 탐색 비율 증가
        exploration_initial_eps=1.0,  # 초기 탐색 확률 
        exploration_final_eps=0.3,  # 최종 탐색 확률 10%
        learning_rate=0.01,  # 학습률 증가
        buffer_size=50000,
        train_freq=1,
        gradient_steps=1,
        batch_size=64,
        gamma=0.99,  # 할인율 증가
        target_update_interval=500,
        )
    # Initialize scenario and tools
    sce = Scenario(vehicleCount=5)
    toolModels = [
        getAvailableActions(env.envs[0]),
        getAvailableLanes(sce),
        getLaneInvolvedCar(sce),
        isChangeLaneConflictWithCar(sce),
        isAccelerationConflictWithCar(sce),
        isKeepSpeedConflictWithCar(sce),
        isDecelerationSafe(sce),
        # isActionSafe()
    ]
    frame = 0
    episode = 0
    
    print("=== 학습 시작 ===")
    for _ in range(10):
        obs = env.reset()
        done = False
        episode_frames = []
        step_count = 0
        
        record_video = (episode % 5 == 0)
        while not done:
            sce.updateVehicles(obs[0], frame)
            # Observation translation
            msg0 = available_action(toolModels)
            msg1 = get_available_lanes(toolModels)
            msg2 = get_involved_cars((toolModels))
            msg1_info = next(iter(msg1.values()))
            lanes_info = extract_lanes_info(msg1_info)

            lane_car_ids = extract_lane_and_car_ids(lanes_info, msg2)
            safety_assessment = assess_lane_change_safety(toolModels, lane_car_ids)
            lane_change_safety = assess_lane_change_safety(toolModels, lane_car_ids)
            safety_msg = check_safety_in_current_lane(toolModels, lane_car_ids)
            formatted_info = format_training_info(msg0, msg1, msg2, lanes_info, lane_car_ids, safety_assessment, lane_change_safety, safety_msg)

            # agent_train.py의 main() 함수 내부 (약 100-110줄 근처)
            action, _ = model.predict(obs)
            action_id = int(action[0])
            action_name = ACTIONS_ALL.get(action_id, "Unknown Action")
            print(f"DQN action: {action_id} -> {action_name}")  # 수정된 디버깅 출력

            llm_response = ask_llm.send_to_chatgpt(action, formatted_info, sce)
            decision_content = llm_response.content
            print(f"LLM action (raw): {decision_content}")  # 추가된 디버깅 출력
            llm_suggested_action = extract_decision(decision_content)
            print(f"LLM action (parsed): {llm_suggested_action}")  # 추가된 디버깅 출력
            print(f"Match: {ACTIONS_ALL.get(action_id) == llm_suggested_action}")  # 추가된 디버깅 출력

            env.env_method('set_llm_suggested_action', llm_suggested_action)

            obs, custom_reward, done, info = env.step(action)
            
            if record_video and step_count % 2 == 0:  # 2스텝마다 저장하여 용량 절약
                frame_img = env.envs[0].render()
                if frame_img is not None:
                    episode_frames.append(frame_img)
            
            print(f"Reward: {custom_reward}\n")
            frame += 1
            step_count += 1
            
            if frame % 10 == 0:  # 10 스텝마다 한 번만 학습
                model.learn(total_timesteps=1, reset_num_timesteps=False)            
                model.save("highway_dqn_model")
            # 나중에 로드하려면:
            # model = DQN.load("highway_dqn_model", env=env)
        if record_video and episode_frames:
            video_filename = f"videos/training_episode_{episode}.mp4"
            save_video_from_frames(episode_frames, video_filename)
        
        print(f"에피소드 {episode} 완료")

    print("=== 학습 완료, 테스트 시작 ===")


    obs = env.reset()
    test_frames = []
    
    for step in range(200):  # 200스텝만 테스트
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # 테스트 중 모든 프레임 저장
        frame_img = env.envs[0].render()
        if frame_img is not None:
            test_frames.append(frame_img)
        
        if dones[0]:
            break
        
        # 시각적 확인을 위한 잠시 대기 (선택사항)
        time.sleep(0.05)
    
    # 테스트 비디오 저장
    if test_frames:
        save_video_from_frames(test_frames, "videos/test_run.mp4", fps=20)
        print("테스트 실행 비디오가 저장되었습니다: videos/test_run.mp4")

    env.close()

def visualize_trained_model():
    """학습된 모델을 실시간으로 시각화"""
    print("=== 실시간 시각화 모드 ===")
    
    # 환경 생성 (실시간 렌더링)
    env = MyHighwayEnv(vehicleCount=5, render_mode='human')
    env = DummyVecEnv([lambda: env])
    
    # 학습된 모델 로드
    try:
        model = DQN.load("highway_dqn_model", env=env)
        print("학습된 모델을 성공적으로 로드했습니다.")
    except:
        print("학습된 모델을 찾을 수 없습니다. 먼저 학습을 진행해주세요.")
        return
    
    # 여러 에피소드 실행
    for episode in range(5):
        obs = env.reset()
        done = False
        step_count = 0
        
        print(f"\n=== 에피소드 {episode + 1} 시작 ===")
        
        while not done and step_count < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            
            action_name = ACTIONS_ALL.get(int(action[0]), "Unknown")
            print(f"Step {step_count}: Action = {action_name}, Reward = {rewards[0]:.2f}")
            
            done = dones[0]
            step_count += 1
            
            # 실시간 시각화를 위한 짧은 대기
            time.sleep(0.1)
        
        print(f"에피소드 {episode + 1} 완료 (총 {step_count}스텝)")
        time.sleep(2)  # 에피소드 간 잠시 대기
    
    env.close()


# utils.py
def extract_decision(response_content):
    try:
        import re
        pattern = r'"decision":\s*{\s*"([^"]+)"\s*}'
        match = re.search(pattern, response_content)
        if match:
            raw_decision = match.group(1).upper().strip()
            
            # 매핑 사전
            decision_map = {
                "ACCELERATE": "FASTER",
                "SPEED UP": "FASTER",
                "GO FASTER": "FASTER",
                "DECELERATE": "SLOWER",
                "SLOW DOWN": "SLOWER",
                "BRAKE": "SLOWER",
                "STAY": "IDLE",
                "MAINTAIN": "IDLE",
                "KEEP": "IDLE",
                "LEFT": "LANE_LEFT",
                "CHANGE LEFT": "LANE_LEFT",
                "RIGHT": "LANE_RIGHT",
                "CHANGE RIGHT": "LANE_RIGHT"
            }
            
            # 매핑 시도
            if raw_decision in ACTIONS_ALL.values():
                return raw_decision  # 이미 올바른 형식
            else:
                mapped = decision_map.get(raw_decision)
                print(f"LLM 응답 '{raw_decision}'을(를) '{mapped}'(으)로 매핑")
                return mapped
        
        print(f"결정 패턴을 찾을 수 없음: {response_content}")
        return None
    except Exception as e:
        print(f"결정 추출 중 오류: {e}")
        return None




if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        # 실시간 시각화 모드
        visualize_trained_model()
    else:
        # 일반 학습 모드
        main()
