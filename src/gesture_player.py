import cv2
import mediapipe as mp
import math
import vlc
import time
import os
import sys
from pathlib import Path

class GestureMediaPlayer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize VLC media player with proper path handling
        if sys.platform.startswith('win32'):
            vlc_paths = [
                'C:\\Program Files\\VideoLAN\\VLC',
                'C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\VideoLAN\\VLC media player.lnk',
                'C:\\Program Files (x86)\\VideoLAN\\VLC'
            ]
            
            # Find VLC installation path
            vlc_path = None
            for path in vlc_paths:
                if os.path.exists(path):
                    vlc_path = path
                    break
            
            if vlc_path is None:
                raise Exception("VLC is not installed in the standard location")

            # Add VLC plugin path to environment
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(vlc_path)
            
            # Set VLC plugin path
            os.environ['PATH'] = vlc_path + ';' + os.environ['PATH']
            plugin_path = os.path.join(vlc_path, 'plugins')
            
            # Initialize VLC instance with plugin path
            self.instance = vlc.Instance('--plugin-path=' + plugin_path)
        else:
            self.instance = vlc.Instance()

        self.player = self.instance.media_player_new()
        self.is_playing = False
        self.volume = 50
        self.player.audio_set_volume(self.volume)

        # Track list and current track index
        self.tracks = []
        self.current_track = 0

        # Gesture states
        self.prev_hand_y = None
        self.prev_hand_x = None
        self.gesture_cooldown = 0
        self.COOLDOWN_TIME = 1.0

    def load_tracks(self, folder_path):
        """Load music tracks from specified folder"""
        self.tracks = list(Path(folder_path).glob('*.mp3'))
        if self.tracks:
            self.load_current_track()

    def load_current_track(self):
        """Load the current track into the player"""
        if 0 <= self.current_track < len(self.tracks):
            media = self.instance.media_new(str(self.tracks[self.current_track]))
            self.player.set_media(media)

    def detect_gestures(self, hand_landmarks):
        # Get hand position and gestures
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Calculate distances
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
        thumb_middle_distance = self.calculate_distance(thumb_tip, middle_tip)

        current_time = time.time()
        if current_time - self.gesture_cooldown < self.COOLDOWN_TIME:
            return

        # Vertical hand movement for volume control
        if self.prev_hand_y is not None:
            y_diff = index_tip.y - self.prev_hand_y
            if abs(y_diff) > 0.02:
                self.volume = max(0, min(100, self.volume - int(y_diff * 100)))
                self.player.audio_set_volume(self.volume)
                self.gesture_cooldown = current_time

        # Horizontal hand movement for track navigation
        if self.prev_hand_x is not None:
            x_diff = index_tip.x - self.prev_hand_x
            if x_diff > 0.1:  # Swipe right
                self.next_track()
                self.gesture_cooldown = current_time
            elif x_diff < -0.1:  # Swipe left
                self.previous_track()
                self.gesture_cooldown = current_time

        # Play/Pause gesture
        if thumb_index_distance < 0.1:
            if self.is_playing:
                self.player.pause()
                self.is_playing = False
            else:
                self.player.play()
                self.is_playing = True
            self.gesture_cooldown = current_time

        self.prev_hand_y = index_tip.y
        self.prev_hand_x = index_tip.x

    def next_track(self):
        """Play next track"""
        if self.tracks:
            self.current_track = (self.current_track + 1) % len(self.tracks)
            self.load_current_track()
            self.player.play()
            self.is_playing = True

    def previous_track(self):
        """Play previous track"""
        if self.tracks:
            self.current_track = (self.current_track - 1) % len(self.tracks)
            self.load_current_track()
            self.player.play()
            self.is_playing = True

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def run(self):
        # Load tracks from music folder
        self.load_tracks("../music_folder")
        
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    self.detect_gestures(hand_landmarks)

            # Display status
            cv2.putText(frame, f"Volume: {self.volume}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Playing: {'Yes' if self.is_playing else 'No'}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.tracks:
                cv2.putText(frame, f"Track: {self.tracks[self.current_track].name}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Gesture Media Player', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    player = GestureMediaPlayer()
    player.run()