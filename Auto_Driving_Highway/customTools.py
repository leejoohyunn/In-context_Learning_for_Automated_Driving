from typing import Any
from scenario import Scenario

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'change lane to the left of the current lane,',
    1: 'remain in the current lane with current speed',
    2: 'change lane to the right of the current lane',
    3: 'accelerate the vehicle',
    4: 'decelerate the vehicle'
}


class getAvailableActions:
    def __init__(self, env: Any) -> None:
        self.env = env

    @prompts(name='Get Available Actions',
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation. The input to this tool should be 'ego'.""")
    def inference(self, input: str) -> str:
        outputPrefix = 'You can ONLY use one of the following actions (exactly as written below, do not substitute with synonyms): \n'
        availableActions = self.env.get_available_actions()
    
        # 사용 가능한 액션을 강조하여 표시
        outputPrefix += "Available actions:\n"
        for action in availableActions:
            action_name = ACTIONS_ALL[action]
            outputPrefix += f"- \"{action_name}\" -- {ACTIONS_DESCRIPTION[action]}\n"
    
        outputPrefix += "\nYour response MUST contain exactly one of these action names in the decision format: \n"
        outputPrefix += "Final Answer: \n    \"decision\": {\"ACTION_NAME_HERE\"},\n"
        outputPrefix += "where ACTION_NAME_HERE is replaced with one of: "
    
        available_action_names = [f"\"{ACTIONS_ALL[action]}\"" for action in availableActions]
        outputPrefix += ", ".join(available_action_names) + ".\n\n"
    
        # 액션 우선순위 설명
        outputPrefix += "ACTION PRIORITY GUIDELINES:\n"
        if 1 in availableActions:
            outputPrefix += "• HIGH PRIORITY: Check IDLE and FASTER actions first.\n"
        if 0 in availableActions or 2 in availableActions:
            outputPrefix += "• MEDIUM PRIORITY: For LANE_LEFT or LANE_RIGHT actions, carefully check the safety of vehicles on target lane.\n"
        if 3 in availableActions:
            outputPrefix += "• MEDIUM PRIORITY: Consider FASTER action when safe.\n"
        if 4 in availableActions:
            outputPrefix += "• LOW PRIORITY: Use SLOWER action only when necessary.\n"
    
        # 안전성 확인 단계
        outputPrefix += "\nTo check decision safety, follow these steps:\n"
        outputPrefix += "1. Identify affected vehicles: FASTER, SLOWER, and IDLE actions affect current lane; LANE_LEFT and LANE_RIGHT affect adjacent lanes.\n"
        outputPrefix += "2. Check safety between ego vehicle and all vehicles in the action lane one by one.\n"
        outputPrefix += "3. If no car is on your current lane, you can safely choose FASTER (but obey traffic rules).\n"
        outputPrefix += "4. For lane changes, check 'Safety Assessment for Lane Changes'; for IDLE, FASTER, or SLOWER, check 'Safety Assessment in Current Lane'.\n"
    
        return outputPrefix


class isActionSafe:
    def __init__(self) -> None:
        pass

    # @prompts(name='Check Action Safety',
    #          description="""Use this tool when you want to check the proposed action's safety. The input to this tool should be a string, which is ONLY the action name.""")
    @prompts(name='Decision-making Instructions',
             description="""This tool gives you a brief intruduction about how to ensure that the action you make is safe. The input to this tool should be a string, which is ONLY the action name.""")
    def inference(self, action: str) -> str:
        return f"""To check action safety you should follow three steps:
        Step 1: Identify the lanes affected by this action. Acceleration, deceleration and idle affect the current lane, while left and right lane changes affect the corresponding lane.
        Step 2:(Optional) Get the vehicles in this lane that you may affect, ONLY when you don't know.
        Step 3: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
        Follow the instructions and remember to use the proper tools mentioned in the tool list once a time.
        """


class getAvailableLanes:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce

    @prompts(name='Get Available Lanes',
             description="""useful when you want to know the available lanes of the vehicles. like: I want to know the available lanes of the vehicle `ego`. The input to this tool should be a string, representing the id of the vehicle.""")
    def inference(self, vid: str) -> str:
        veh = self.sce.vehicles[vid]
        raw = veh.lane_id
        currentLaneID = raw if isinstance(raw, str) else f"lane_{raw}"
    
       # 동적으로 유효한 레인 확인
        valid_lanes = list(self.sce.lanes.keys())
        if currentLaneID not in valid_lanes:
            print(f"[WARNING getAvailableLanes] 잘못된 레인: {currentLaneID}, 유효한 레인: {valid_lanes}")
            # 가장 가까운 레인으로 수정
            try:
                lane_num = int(currentLaneID.split('_')[1]) if '_' in currentLaneID else 1
                lane_num = max(0, min(lane_num, len(valid_lanes) - 1))
                currentLaneID = f"lane_{lane_num}"
                veh.lane_id = currentLaneID  # ⭐ 차량 정보도 업데이트
                print(f"[WARNING] 차량 {vid}의 레인을 {currentLaneID}로 수정")
            except:
                currentLaneID = 'lane_1'
                veh.lane_id = currentLaneID
    
        #동적 레인 인덱스 계산
        if currentLaneID not in self.sce.lanes:
            return f"Error: lane '{currentLaneID}' not found"

        laneIdx = self.sce.lanes[currentLaneID].laneIdx
        max_lane_idx = max(lane.laneIdx for lane in self.sce.lanes.values())
        min_lane_idx = min(lane.laneIdx for lane in self.sce.lanes.values())
    
        # 동적으로 인접 레인 계산
        available_lanes = []
        lane_descriptions = []
    
        # 현재 레인
        available_lanes.append((currentLaneID, "current"))
    
        # 왼쪽 레인 (인덱스가 더 작은 레인)
        if laneIdx > min_lane_idx:
            leftLane = f"lane_{laneIdx-1}"
            if leftLane in valid_lanes:
                available_lanes.append((leftLane, "left"))
                lane_descriptions.append(f"`{leftLane}` is to the left")
    
        # 오른쪽 레인 (인덱스가 더 큰 레인)
        if laneIdx < max_lane_idx:
            rightLane = f"lane_{laneIdx+1}"
            if rightLane in valid_lanes:
                available_lanes.append((rightLane, "right"))
                lane_descriptions.append(f"`{rightLane}` is to the right")
    
        # 응답 구성
        lane_ids = [f"`{lane[0]}` ({lane[1]})" for lane in available_lanes]
        response = f"The available lanes for `{vid}` are {', '.join(lane_ids)}. "
        response += f"`{currentLaneID}` is the current lane"
        if lane_descriptions:
            response += "; " + "; ".join(lane_descriptions)
        response += "."
    
        return response

class getLaneInvolvedCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce

    @prompts(name='Get Lane Involved Car',
             description="""useful whent want to know the cars may affect your action in the certain lane. Make sure you have use tool `Get Available Lanes` first. The input is a string, representing the id of the specific lane you want to drive on, DONNOT input multiple lane_id once.""")
    def inference(self, laneID: str) -> str:
        valid_lanes = list(self.sce.lanes.keys())  # 실제 존재하는 레인 동적 확인
        if not valid_lanes:  # 비어있다면 기본값 사용
            valid_lanes = ['lane_0', 'lane_1', 'lane_2']
            
        if laneID not in valid_lanes:
            print(f"[WARNING] 잘못된 레인 ID: {laneID}, 유효한 레인: {valid_lanes}")
        # ⭐ 수정된 부분: 가장 가까운 유효한 레인으로 자동 수정
            try:
                lane_num = int(laneID.split('_')[1]) if '_' in laneID else 1
                lane_num = max(0, min(lane_num, len(valid_lanes) - 1))  # 범위 내로 제한
                laneID = f"lane_{lane_num}"
                print(f"[WARNING] 자동 수정된 레인 ID: {laneID}")
            except:
                laneID = 'lane_1'  # 기본값
                print(f"[WARNING] 기본 레인으로 설정: {laneID}")
    
        ego = self.sce.vehicles['ego']
        laneVehicles = []
        
        for vk, vv in self.sce.vehicles.items():
            if vk != 'ego' and hasattr(vv, 'presence') and vv.presence:
                if vv.lane_id == laneID:
                    laneVehicles.append((vv.id, vv.lanePosition))
                    
        laneVehicles.sort(key=lambda x: x[1])
        leadingCarIdx = -1
        
        for i in range(len(laneVehicles)):
            vp = laneVehicles[i]
            if vp[1] >= ego.lanePosition:
                leadingCarIdx = i
                break
        if leadingCarIdx == -1:
            if not laneVehicles:  # ⭐ 수정된 부분: 명시적 빈 리스트 확인
                return f'There is no car driving on {laneID}. This lane is safe, you do not need to check for any vehicle for safety! You can drive on this lane safely.'
            else:
                rearingCar = laneVehicles[-1][0]
                return f"{rearingCar} is driving on {laneID}, and it's driving behind ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        elif leadingCarIdx == 0:
            leadingCar = laneVehicles[0][0]
            distance = round(laneVehicles[0][1] - ego.lanePosition, 2)
            leading_car_vel = round(self.sce.vehicles[leadingCar].speed,1)
            return f"{leadingCar} is driving at {leading_car_vel}m/s on {laneID}, and it's driving in front of ego car for {distance} meters. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        else:
            leadingCar = laneVehicles[leadingCarIdx][0]
            rearingCar = laneVehicles[leadingCarIdx-1][0]
            distance = round(laneVehicles[leadingCarIdx][1] - ego.lanePosition, 2)
            leading_car_vel = round(self.sce.vehicles[leadingCar].speed,1)
            return f"{leadingCar} and {rearingCar} is driving on {laneID}, and {leadingCar} is driving at {leading_car_vel}m/s in front of ego car for {distance} meters, while {rearingCar} is driving behind ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."


class isChangeLaneConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0

    @prompts(name='Is Change Lane Confict With Car',
             description="""useful when you want to know whether change lane to a specific lane is confict with a specific car, ONLY when your decision is change_lane_left or change_lane_right. The input to this tool should be a string of a comma separated string of two, representing the id of the lane you want to change to and the id of the car you want to check.""")
    def inference(self, inputs: str) -> str:
        laneID, vid = inputs.replace(' ', '').split(',')
        if vid not in self.sce.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']
        if veh.lanePosition >= ego.lanePosition:
            relativeSpeed = ego.speed - veh.speed
            if veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"change lane to `{laneID}` is safe with `{vid}`."
            else:
                return f"change lane to `{laneID}` may be conflict with `{vid}`, which is unacceptable."
        else:
            relativeSpeed = veh.speed - ego.speed
            if ego.lanePosition - veh.lanePosition - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"change lane to `{laneID}` is safe with `{vid}`."
            else:
                return f"change lane to `{laneID}` may be conflict with `{vid}`, which is unacceptable."


class isAccelerationConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0
        self.acceleration = 4.0

    @prompts(name='Is Acceleration Conflict With Car',
             description="""useful when you want to know whether acceleration is safe with a specific car, ONLY when your decision is accelerate. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']
        if veh.lane_id != ego.lane_id:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_id == ego.lane_id:
            if veh.lanePosition >= ego.lanePosition:
                relativeSpeed = ego.speed + self.acceleration - veh.speed
                distance = veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH * 2
                if distance > self.TIME_HEAD_WAY * relativeSpeed:
                    return f"acceleration is safe with `{vid}`."
                else:
                    return f"acceleration may be conflict with `{vid}`, which is unacceptable."
            else:
                return f"acceleration is safe with {vid}"
        else:
            return f"acceleration is safe with {vid}"


class isKeepSpeedConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0

    @prompts(name='Is Keep Speed Conflict With Car',
             description="""useful when you want to know whether keep speed is safe with a specific car, ONLY when your decision is keep_speed. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']
        if veh.lane_id != ego.lane_id:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_id == ego.lane_id:
            if veh.lanePosition >= ego.lanePosition:
                relativeSpeed = ego.speed - veh.speed
                distance = veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH * 2
                if distance > self.TIME_HEAD_WAY * relativeSpeed:
                    return f"keep lane with current speed is safe with {vid}"
                else:
                    return f"keep lane with current speed may be conflict with {vid}, you need consider decelerate"
            else:
                return f"keep lane with current speed is safe with {vid}"
        else:
            return f"keep lane with current speed is safe with {vid}"


class isDecelerationSafe:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0
        self.deceleration = 3.0

    @prompts(name='Is Deceleration Safe',
             description="""useful when you want to know whether deceleration is safe, ONLY when your decision is decelerate.The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, vid: str) -> str:
        if vid not in self.sce.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        veh = self.sce.vehicles[vid]
        ego = self.sce.vehicles['ego']
        if veh.lane_id != ego.lane_id:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_id == ego.lane_id:
            if veh.lanePosition >= ego.lanePosition:
                relativeSpeed = ego.speed - veh.speed - self.deceleration
                distance = veh.lanePosition - ego.lanePosition - self.VEHICLE_LENGTH
                if distance > self.TIME_HEAD_WAY * relativeSpeed:
                    return f"deceleration with current speed is safe with {vid}"
                else:
                    return f"deceleration with current speed may be conflict with {vid}, if you have no other choice, slow down as much as possible"
            else:
                return f"deceleration with current speed is safe with {vid}"
        else:
            return f"deceleration with current speed is safe with {vid}"
