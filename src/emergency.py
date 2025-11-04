class EmergencyHandler:
    def __init__(self):
        self.active = False
        self.last_trigger_time = 0
        self.cooldown_period = 5.0  # seconds
        
    def trigger_emergency(self, player):
        """Handle emergency stop"""
        self.active = True
        player.player.stop()
        player.volume = 0
        player.player.audio_set_volume(0)
        player.is_playing = False
        return True
        
    def reset(self):
        """Reset emergency state"""
        self.active = False