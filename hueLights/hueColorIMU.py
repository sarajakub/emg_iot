"""
EMG + IMU-controlled Philips Hue Light Color Adjustment

Control Hue light color using EMG gestures + IMU motion from a Myo armband.
Hold a gesture and move your arm left/right to decrease/increase color hue smoothly.

SETUP:
    BRIDGE SETUP:
    - Change BRIDGE_IP to your Hue Bridge IP address (found in Hue app settings).
    - Ensure computer is on same wifi network as Hue Bridge.
    - Press button on Hue Bridge before running for first time.

    MYO ARMBAND SETUP:
    - Ensure Myo dongle is connected and armband is on.
    - Position Myo with port facing hand, light facing towards the ceiling (on top of arm).

    Requirements: pyomyo, phue, xgboost, pygame
    Install: pip install pyomyo phue xgboost pygame

USAGE:
    - Set TRAINING_MODE = True to train gesture recognition
    - Set TRAINING_MODE = False for color/hue control
    - In training: Press '0' for rest, '1' for gesture (e.g. fist)
    - In control: Hold gesture + move arm left/right to adjust color/hue
"""

from phue import Bridge
from pyomyo import emg_mode
from pyomyo.Classifier import Live_Classifier, MyoClassifier, EMGHandler
from xgboost import XGBClassifier
import pygame
import time
import os

# === CONFIGURATION ===
TRAINING_MODE = False  # True for training, False for control
BRIDGE_IP = '192.168.1.89'  # Replace with your Hue Bridge IP

# Light control options (choose one):
# LIGHT_NAMES = ['Back Right', 'Back Left']  # For individual lights, replace with your light names
GROUP_NAME = 'Living room'                 # For Hue groups, replace with your group name (Case-sensitive)
USE_GROUP = True                         # True for groups, False for individual lights

# Control parameters
UPDATE_INTERVAL = 0.3  # Seconds between hue updates (slower, more gradual changes)
MOTION_THRESHOLD = 150  # Minimum IMU change to trigger hue adjustment (more sensitive)
HUE_STEP = 500  # Hue change per motion unit (smaller steps for gradual changes)
DEADZONE = 50  # IMU deadzone to prevent jitter (smaller for more responsiveness)

# Global variables
current_gesture = 0
imu_y_value = 0
last_hue_update = 0
gesture_start_y = 0  # Track starting Y position when gesture begins

# === INITIALIZATION ===
def setup_hue_bridge():
    """Initialize Hue Bridge connection and display available devices."""
    try:
        bridge = Bridge(BRIDGE_IP)
        bridge.connect()
        
        if USE_GROUP:
            groups = bridge.get_group()
            available = [g['name'] for g in groups.values()]
            print(f"Connected to Hue Bridge. Available groups: {available}")
        else:
            lights = bridge.get_light_objects('name')
            available = list(lights.keys())
            print(f"Connected to Hue Bridge. Available lights: {available}")
            
        return bridge
    except Exception as e:
        print(f"[ERROR] Could not connect to Hue Bridge: {e}")
        return None


def adjust_hue(bridge, hue_change):
    """Adjust hue value for lights or group based on IMU movement."""
    if not bridge:
        return False
        
    try:
        if USE_GROUP:
            groups = bridge.get_group()
            group_id = next((gid for gid, g in groups.items() if g['name'] == GROUP_NAME), None)
            
            if group_id:
                # Get current hue
                current_state = groups[group_id]['action']
                current_hue = current_state.get('hue', 0)
                
                # Calculate new hue (wrap around at 65535)
                new_hue = (current_hue + hue_change) % 65536
                new_hue = max(0, min(65535, new_hue))
                
                # Update group hue
                bridge.set_group(int(group_id), 'hue', new_hue)
                print(f"Group '{GROUP_NAME}' hue: {current_hue} -> {new_hue}")
                return True
            else:
                print(f"Group '{GROUP_NAME}' not found")
                return False
        else:
            lights = bridge.get_light_objects('name')
            available_lights = [name for name in LIGHT_NAMES if name in lights]
            
            if available_lights:
                # Get current hue from first light (assume all lights have same hue)
                current_hue = lights[available_lights[0]].hue
                
                # Calculate new hue (wrap around at 65535)
                new_hue = (current_hue + hue_change) % 65536
                new_hue = max(0, min(65535, new_hue))
                
                # Update all specified lights
                for name in available_lights:
                    lights[name].hue = new_hue
                print(f"Lights {available_lights} hue: {current_hue} -> {new_hue}")
                return True
            else:
                print("No specified lights found")
                return False
                
    except Exception as e:
        print(f"Error adjusting hue: {e}")
        return False


def handle_gesture(pose, bridge):
    """Process detected EMG gestures."""
    global current_gesture, gesture_start_y, imu_y_value
    if pose != current_gesture:
        current_gesture = pose
        if pose == 1:
            # Capture starting position when gesture begins
            gesture_start_y = imu_y_value
            print(f"Gesture active - hue control enabled (starting Y: {gesture_start_y:.1f})")
        else:
            print("Gesture inactive - hue control disabled")


def handle_imu(quat, acc, gyro, bridge):
    """Process IMU data for motion detection."""
    global imu_y_value, last_hue_update, current_gesture, gesture_start_y
    
    # Always update current IMU value
    imu_y_value = acc[1]
    
    # Only process IMU when gesture is active and not in training mode
    if current_gesture != 1 or TRAINING_MODE:
        return
    
    # Calculate relative movement from starting position
    relative_y = imu_y_value - gesture_start_y
    
    # Rate limiting for bridge protection
    current_time = time.time()
    if current_time - last_hue_update < UPDATE_INTERVAL:
        return
    
    # Apply deadzone to prevent jitter
    if abs(relative_y) < DEADZONE:
        return
    
    # Calculate hue change based on relative IMU movement
    if abs(relative_y) > MOTION_THRESHOLD:
        hue_change = int((relative_y / MOTION_THRESHOLD) * HUE_STEP)
        
        if adjust_hue(bridge, hue_change):
            last_hue_update = current_time
            print(f"Relative Y: {relative_y:.1f} (from {gesture_start_y:.1f}) -> Hue change: {hue_change}")


def main():
    """Main application entry point."""
    
    print("Initializing Hue IMU Control...")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Initialize Hue Bridge
    bridge = setup_hue_bridge()
    
    # Setup training interface if needed
    if TRAINING_MODE:
        print("TRAINING MODE: Press 0 (rest) and 1 (fist) to train gestures")
        pygame.init()
        screen = pygame.display.set_mode((800, 320))
        font = pygame.font.Font(None, 30)
        pygame.display.set_caption("EMG Hue Control - Training")
    else:
        target = f"group '{GROUP_NAME}'" if USE_GROUP else f"lights {LIGHT_NAMES}"
        print(f"CONTROL MODE: Hold fist gesture + move arm left/right to adjust hue for {target}")
    
    # Initialize EMG classifier
    try:
        model = XGBClassifier(
            eval_metric='logloss',
            objective='binary:logistic',
            base_score=0.5,
            random_state=42,
            n_estimators=50
        )
        classifier = Live_Classifier(model, name="HueIMU", color=(255, 100, 50))
        myo = MyoClassifier(classifier, mode=emg_mode.PREPROCESSED, hist_len=10)
        
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        print("You may need to train gestures first or clear corrupted data")
        return
    
    # Setup training handler if in training mode
    if TRAINING_MODE:
        emg_handler = EMGHandler(myo)
        myo.add_emg_handler(emg_handler)
    
    # Add gesture and IMU handlers (pass bridge to handlers)
    myo.add_raw_pose_handler(lambda pose: handle_gesture(pose, bridge))
    myo.add_imu_handler(lambda quat, acc, gyro: handle_imu(quat, acc, gyro, bridge))
    
    # Main execution loop
    try:
        myo.connect()
        print("Connected to Myo armband")
        print("Ready for operation...")
        
        while True:
            myo.run()
            if TRAINING_MODE:
                myo.run_gui(emg_handler, screen, font, 800, 320)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        myo.disconnect()
        if TRAINING_MODE:
            pygame.quit()
        print("Disconnected")


if __name__ == '__main__':
    main()
