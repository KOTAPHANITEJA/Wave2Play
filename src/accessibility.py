class AccessibilityMode:
    def __init__(self):
        self.enabled = False
        self.sensitivity = 1.0
        self.cooldown_time = 2.0
        self.single_finger_mode = False
        
    def adjust_sensitivity(self, mode):
        if mode == "limited_mobility":
            self.sensitivity = 0.7
            self.cooldown_time = 3.0
            self.single_finger_mode = True
        else:
            self.sensitivity = 1.0
            self.cooldown_time = 2.0
            self.single_finger_mode = False