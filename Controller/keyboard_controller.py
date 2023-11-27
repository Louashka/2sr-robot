class ActionsHandler:

    def __init__(self, omni_speed, rotation_speed, lu_speed) -> None:

        self.__omni_speed = omni_speed  # Agent velocity in the rigid mode
        self.__rotation_speed = rotation_speed # Agent angular velocity in the rigid mode
        self.__lu_speed = lu_speed  # LU's velocity in the soft mode

        self.__current_keys = set() # Set of pressed keys

        self.__v = [0] * 5 # Array of control velocities
        self.__s = [0, 0] # Values of VS segments stiffness 

        # Control key combinations
        self.__keyboard_combinations = {
            'FORWARD': {'Up'},
            'BACKWARD': {'Down'},
            'RIGHT': {'Right'},
            'LEFT': {'Left'},
            'ROTATION_LEFT': {'w', 'Left'},
            'ROTATION_RIGHT': {'w', 'Right'},
            'HEAD_FORWARD': {'a', 'Up'},
            'HEAD_BACKWARD': {'a', 'Down'},
            'TAIL_FORWARD': {'d', 'Up'},
            'TAIL_BACKWARD': {'d', 'Down'},
            'HT_FORWARD': {'a', 'd', 'Up'},
            'HT_BACKWARD': {'a', 'd', 'Down'},
            'S1_SOFT': {'a', 's', 'Up'},
            'S1_RIGID': {'a', 's', 'Down'},
            'S2_SOFT': {'d', 's', 'Up'},
            'S2_RIGID': {'d', 's', 'Down'},
            'BOTH_SOFT': {'s', 'Up'},
            'BOTH_RIGID': {'s', 'Down'}
        }

        # Actions that correspond to the key combinations
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
            # Define control velocities and stiffness values according to pressed key combination
            if (keys == self.__current_keys):
                action = self.__keyboard_actions[combination]['callback']
                message = self.__keyboard_actions[combination]['action']

                print(message) # Print the action message
                action()
                break
            else:
                # Stop the motion if necessary keys are not pressed
                self.__v = [0] * 5

    def onPress(self, key):
        self.__current_keys.add(key.keysym)
        self.__handleKeyCombination()

    def onRelease(self, key):
        try:
            self.__current_keys.remove(key.keysym)
        except KeyError:
            self.__current_keys.clear()

        self.__handleKeyCombination()

    @property
    def v(self):
        return self.__v

    @property
    def s(self):
        return self.__s
