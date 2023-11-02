import manual_motion
from pynput import keyboard


class FK(manual_motion.KeyboardController):

    def __init__(self, omni_speed, rotation_speed, lu_speed) -> None:
        super().__init__(omni_speed, rotation_speed, lu_speed)

    def onPress(self, key):
        v, s = super().onPress(key)

    def onRelease(self, key):
        v, s = super().onRelease(key)


if __name__ == "__main__":
    mm = FK(0.1, 1, 0.1)
    with keyboard.Listener(on_press=mm.onPress, on_release=mm.onRelease) as listener:
        listener.join()
