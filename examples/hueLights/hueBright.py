"""
EMG-controlled Philips Hue Light Toggle

Control Hue lights or groups using EMG gestures from a Myo armband.

SETUP
    BRIDGE SETUP:
    - Change BRIDGE_IP to your Hue Bridge IP address (found in the settings of the Hue app).
    - Ensure your computer is on the same wifi network as the Hue Bridge.
    - Press button on Hue Bridge to allow connections right before running the script for the first time.

    MYO ARMBAND SETUP:
    - Make sure your Myo armband dongle is connected and the Myo armband is on and recognized.
    - Make sure Myo armband port is facing hand and the Myo light is showing on the top of your arm 
        - See pyomyo library for more details: https://github.com/PerlinWarp/pyomyo
    
    Requirements: pyomyo, phue, xgboost, pygame
    - Install the required packages: `pip install phue pyomyo xgboost pygame` or `pip install -r requirements.txt`.


Training mode allows you to train gestures, while control mode operates the lights. 
TRAINING_MODE for gesture training (signal detection):
    - Supports both training mode for gesture recognition and control mode for light operation. 
    - Switch between modes by setting TRAINING_MODE to True or False. 
    - Press '0' when resting arm and '1' for trained gesture (fist) in training mode.
    - In control mode, make the trained gesture (e.g. fist) to toggle lights or groups on/off.
    - Ensure the 'data' directory exists for storing training data.


Light control options:
    - Change the BRIDGE_IP to your Hue Bridge IP address (case-sensitive).
    - Change GROUP_NAME to your Hue group name or LIGHT_NAMES for individual lights (case-sensitive).
    - Set USE_GROUP to True for groups, False for individual lights.
    - Comment and uncomment the appropriate sections for light control.
"""

from phue import Bridge
from pyomyo import emg_mode
from pyomyo.Classifier import Live_Classifier, MyoClassifier, EMGHandler
from xgboost import XGBClassifier
import pygame
from pygame.locals import *
import os

# Configuration
TRAINING_MODE = True # Set to True to enable training EMG mode; set to False for control mode
BRIDGE_IP = '192.168.1.01' # Replace with your Hue Bridge IP (Found in the settings of Hue app)
os.makedirs('data', exist_ok=True) # Ensure the 'data' directory exists for storing training data

# Light control options (choose one):

# LIGHT_NAMES = ['Back Right', 'Back Left']  # For individual lights, replace with your light names
GROUP_NAME = 'Living room'                 # For Hue groups, replace with your group name (Case-sensitive)
USE_GROUP = True                         # True for groups, False for individual lights


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


def toggle_lights(bridge):
    """Toggle lights or group on/off based on current state."""
    if not bridge:
        return
        
    try:
        if USE_GROUP:
            groups = bridge.get_group()
            group_id = next((gid for gid, g in groups.items() if g['name'] == GROUP_NAME), None)
            
            if group_id:
                current_state = groups[group_id]['state']['any_on']
                bridge.set_group(int(group_id), 'on', not current_state)
                print(f"Group '{GROUP_NAME}' {'OFF' if current_state else 'ON'}")
            else:
                print(f"Group '{GROUP_NAME}' not found")
        else:
            lights = bridge.get_light_objects('name')
            available_lights = [name for name in LIGHT_NAMES if name in lights]
            
            if available_lights:
                current_state = lights[available_lights[0]].on
                for name in available_lights:
                    lights[name].on = not current_state
                print(f"Lights {available_lights} {'OFF' if current_state else 'ON'}")
            else:
                print("No specified lights found")
                
    except Exception as e:
        print(f"Error toggling lights: {e}")


def handle_gesture(pose, bridge):
    """Process detected EMG gestures."""
    if pose == 1 and not TRAINING_MODE:
        toggle_lights(bridge)


def main():
    """Main application entry point."""
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
        pygame.display.set_caption("EMG Light Control - Training")
    else:
        target = f"group '{GROUP_NAME}'" if USE_GROUP else f"lights {LIGHT_NAMES}"
        print(f"CONTROL MODE: Make a fist to toggle {target}")
    
    # Initialize EMG classifier
    model = XGBClassifier(eval_metric='logloss', base_score=0.5, objective='binary:logistic')
    classifier = Live_Classifier(model, name="LightToggle", color=(50, 150, 255))
    myo = MyoClassifier(classifier, mode=emg_mode.PREPROCESSED, hist_len=10)
    
    # Setup training handler if in training mode
    if TRAINING_MODE:
        emg_handler = EMGHandler(myo)
        myo.add_emg_handler(emg_handler)
    
    # Add gesture handler
    myo.add_raw_pose_handler(lambda pose: handle_gesture(pose, bridge))
    
    try:
        myo.connect()
        print("Connected to Myo armband")
        
        # Main loop
        while True:
            myo.run()
            if TRAINING_MODE:
                myo.run_gui(emg_handler, screen, font, 800, 320)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        myo.disconnect()
        if TRAINING_MODE:
            pygame.quit()
        print("Disconnected")


if __name__ == '__main__':
    main()
