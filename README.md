# emg_iot
Controlling IOT devices with the Myo EMG armband (muscle signals)
- Philips Hue Lights [`hueLights`](https://github.com/sarajakub/emg_iot/tree/main/examples/hueLights)

## Controlling Philips Hue Lights with EMG
![myolightio_github](https://github.com/user-attachments/assets/97ff81b7-47d7-4693-9fd1-529c9c55aa00)

Built using [`pyomyo`](https://github.com/PerlinWarp/pyomyo) and [`phue`](https://github.com/studioimaginaire/phue)

- Toggle Hue lights or groups with a trained gesture
- Enter gesture training mode, collect custom EMG data, switch out of training to control lights
  - Modular design for expanding gesture mappings

#### What's needed

- Python
- Myo armband
- Philips Hue Bridge + lights

Directions on how to use Myo EMG armband and Philips Hue Lights: [`hueBright`](https://github.com/sarajakub/emg_iot/blob/main/examples/hueLights/hueBright.py)

- More info on Myo setup: [`pyomyo`](https://github.com/PerlinWarp/pyomyo)
- More info on Philips Hue setup: [`phue`](https://github.com/studioimaginaire/phue)

#### Python Dependencies
Install from `requirements.txt`:

```bash
pip install -r requirements.txt
