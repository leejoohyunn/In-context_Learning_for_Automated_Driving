#!/usr/bin/env python3
"""
차량 시뮬레이션 시각화 실행 스크립트

사용법:
1. 학습 + 비디오 저장: python run_visualization.py
2. 실시간 시각화: python run_visualization.py --live
3. 비디오만 생성: python run_visualization.py --video-only
"""

import sys
import os
import argparse
from agent_train import main, visualize_trained_model
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import time

def create_video_only():
    """이미 학습된 모델로 비디오만 생성"""
    print("=== 비디오 생성 모드 ===")
    
    from agent_train import MyHighwayEnv, save_video_from_frames, ACTIONS_ALL
    
    # 환경 생성
    env = MyHighwayEnv(vehicleCount=5, render_mode='rgb_array')
    env = DummyVecEnv([lambda: env])
    
    # 학습된 모델 로드
    try:
        model = DQN.load("highway_dqn_model", env=env)
        print("학습된 모델을 성공적으로 로드했습니다.")
    except:
        print("❌ 학습된 모델을 찾을 수 없습니다. 먼저 학습을 진행해주세요.")
        return
    
    os.makedirs("videos", exist_ok=True)
    
    # 여러 에피소드 비디오 생성
    for episode in range(3):
        print(f"에피소드 {episode + 1} 비디오 생성 중...")
        obs = env.reset()
        done = False
        frames = []
        step_count = 0
        
        while not done and step_count < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            
            # 프레임 저장
            frame_img = env.envs[0].render()
            if frame_img is not None:
                frames.append(frame_img)
            
            done = dones[0]
            step_count += 1
        
        # 비디오 저장
        if frames:
            filename = f"videos/episode_{episode + 1}_steps_{step_count}.mp4"
            save_video_from_frames(frames, filename, fps=15)
    
    print("✅ 모든 비디오 생성 완료! 'videos' 폴더를 확인하세요.")
    env.close()

def main_cli():
    parser = argparse.ArgumentParser(description='차량 시뮬레이션 시각화')
    parser.add_argument('--live', action='store_true', 
                       help='실시간 시각화 모드 (학습된 모델 필요)')
    parser.add_argument('--video-only', action='store_true',
                       help='비디오만 생성 (학습된 모델 필요)')
    
    args = parser.parse_args()
    
    if args.live:
        visualize_trained_model()
    elif args.video_only:
        create_video_only()
    else:
        # 기본 학습 모드
        print("=== 학습 + 비디오 저장 모드 ===")
        print("학습 중 5에피소드마다 비디오가 저장됩니다.")
        print("완료 후 'videos' 폴더에서 결과를 확인하세요.\n")
        main()

if __name__ == "__main__":
    main_cli()
