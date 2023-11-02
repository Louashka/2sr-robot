from pynput import keyboard


class ActionsHandler:

    def __init__(self, omni_speed, rotation_speed, lu_speed) -> None:

        self.__omni_speed = omni_speed  # Speed of the agent in the rigid mode
        # Speed of the agent's rotation in the rigid mode
        self.__rotation_speed = rotation_speed
        self.__lu_speed = lu_speed  # Speed of the locomotion units int thxe soft mode

        self.__current_keys = set()

        self.__v = [0] * 5
        self.__s = [0, 0]

        # Key combinations
        self.__keyboard_combinations = {
            'FORWARD': {keyboard.Key.up},
            'BACKWARD': {keyboard.Key.down},
            'RIGHT': {keyboard.Key.right},
            'LEFT': {keyboard.Key.left},
            'ROTATION_LEFT': {keyboard.KeyCode.from_char('w'), keyboard.Key.left},
            'ROTATION_RIGHT': {keyboard.KeyCode.from_char('w'), keyboard.Key.right},
            'HEAD_FORWARD': {keyboard.KeyCode.from_char('a'), keyboard.Key.up},
            'HEAD_BACKWARD': {keyboard.KeyCode.from_char('a'), keyboard.Key.down},
            'TAIL_FORWARD': {keyboard.KeyCode.from_char('d'), keyboard.Key.up},
            'TAIL_BACKWARD': {keyboard.KeyCode.from_char('d'), keyboard.Key.down},
            'HT_FORWARD': {keyboard.KeyCode.from_char('a'), keyboard.KeyCode.from_char('d'), keyboard.Key.up},
            'HT_BACKWARD': {keyboard.KeyCode.from_char('a'), keyboard.KeyCode.from_char('d'), keyboard.Key.down},
            'S1_SOFT': {keyboard.KeyCode.from_char('a'), keyboard.KeyCode.from_char('s'), keyboard.Key.up},
            'S1_RIGID': {keyboard.KeyCode.from_char('a'), keyboard.KeyCode.from_char('s'), keyboard.Key.down},
            'S2_SOFT': {keyboard.KeyCode.from_char('d'), keyboard.KeyCode.from_char('s'), keyboard.Key.up},
            'S2_RIGID': {keyboard.KeyCode.from_char('d'), keyboard.KeyCode.from_char('s'), keyboard.Key.down},
            'BOTH_SOFT': {keyboard.KeyCode.from_char('s'), keyboard.Key.up},
            'BOTH_RIGID': {keyboard.KeyCode.from_char('s'), keyboard.Key.down}
        }

        self.__keyboard_actions = {
            'FORWARD': {
                'action': 'Agent forward',
                'callback': lambda: self.__v.__setitem__(3, self.__omni_speed)
            },
            'BACKWARD': {
                'action': 'Agent backward',
                'callback': lambda: self.__v.__setitem__(3, -self.__omni_speed)
            },
            'RIGHT': {
                'action': 'Agent right',
                'callback': lambda: self.__v.__setitem__(2, self.__omni_speed)
            },
            'LEFT': {
                'action': 'Agent left',
                'callback': lambda: self.__v.__setitem__(2, -self.__omni_speed)
            },
            'ROTATION_RIGHT': {
                'action': 'Turn agent to the right',
                'callback': lambda: self.__v.__setitem__(4, -self.__rotation_speed)
            },
            'ROTATION_LEFT': {
                'action': 'Turn agent to the left',
                'callback': lambda: self.__v.__setitem__(4, self.__rotation_speed)
            },
            'HEAD_FORWARD': {
                'action': 'Head LU forward',
                'callback': lambda: self.__v.__setitem__(0, self.__lu_speed)
            },
            'HEAD_BACKWARD': {
                'action': 'Head LU backward',
                'callback': lambda: self.__v.__setitem__(0, -self.__lu_speed)
            },
            'TAIL_FORWARD': {
                'action': 'Tail LU forward',
                'callback': lambda: self.__v.__setitem__(1, self.__lu_speed)
            },
            'TAIL_BACKWARD': {
                'action': 'Tail LU backward',
                'callback': lambda: self.__v.__setitem__(1, -self.__lu_speed)
            },
            'HT_FORWARD': {
                'action': 'Both LU\'s forward',
                'callback': lambda: {self.__v.__setitem__(0, self.__lu_speed), self.__v.__setitem__(1, self.__lu_speed)}
            },
            'HT_BACKWARD': {
                'action': 'Both LU\'s backward',
                'callback': lambda: {self.__v.__setitem__(0, -self.__lu_speed), self.__v.__setitem__(1, -self.__lu_speed)}
            },
            'S1_SOFT': {
                'action': 'Switch a head VSS to soft',
                'callback': lambda: self.__s.__setitem__(0, 1)
            },
            'S1_RIGID': {
                'action': 'Switch a head VSS to rigid',
                'callback': lambda: self.__s.__setitem__(0, 0)
            },
            'S2_SOFT': {
                'action': 'Switch a tail VSS to the soft mode',
                'callback': lambda: self.__s.__setitem__(1, 1)
            },
            'S2_RIGID': {
                'action': 'Switch a tail VSS to the rigid mode',
                'callback': lambda: self.__s.__setitem__(1, 0)
            },
            'BOTH_SOFT': {
                'action': 'Switch both VSS to the soft mode',
                'callback': lambda: {self.__s.__setitem__(0, 1), self.__s.__setitem__(1, 1)}
            },
            'BOTH_RIGID': {
                'action': 'Switch both VSS to the rigid mode',
                'callback': lambda: {self.__s.__setitem__(0, 0), self.__s.__setitem__(1, 0)}
            },
        }

    def __handleKeyCombination(self):
        for combination, keys in self.__keyboard_combinations.items():
            if (keys == self.__current_keys):
                action = self.__keyboard_actions[combination]["callback"]
                message = self.__keyboard_actions[combination]["action"]

                print(message)
                action()
                break
            else:
                self.__v = [0] * 5

    def onPress(self, key):
        self.__current_keys.add(key)

        self.__handleKeyCombination()

    def onRelease(self, key):
        try:
            self.__current_keys.remove(key)
        except KeyError:
            self.__current_keys.clear()

        self.__handleKeyCombination()

    @property
    def v(self):
        return self.__v

    @property
    def s(self):
        return self.__s
