

class MotionPlanner:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal

    def is_directly_reachable(self, state):
        # Check if the given state is within the reachable workspace
        # This depends on your specific robot's constraints
        pass

if __name__ == '__main__':
    pass