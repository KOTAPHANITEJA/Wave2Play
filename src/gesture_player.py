import cv2
import mediapipe as mp
import math
import vlc
import time
import os
import sys
import ctypes
from pathlib import Path
from src.accessibility import AccessibilityMode
from src.web.server import WebInterface
from src.emergency import EmergencyHandler

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

        # Initialize state variables first
        self.volume = 50
        self.is_playing = False
        self.is_muted = False
        self.is_video = False
        self.window_name = 'Media Playback'

        # Initialize VLC here, before using any VLC functions
        try:
            if sys.platform.startswith('win32'):
                vlc_paths = [
                    'C:\\Program Files\\VideoLAN\\VLC',
                    'C:\\Program Files (x86)\\VideoLAN\\VLC'
                ]
                
                vlc_path = next((path for path in vlc_paths if os.path.exists(path)), None)
                
                if vlc_path is None:
                    print("VLC not found. Please install VLC media player.")
                    sys.exit(1)

                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(vlc_path)
                
                os.environ['PATH'] = vlc_path + ';' + os.environ['PATH']
                plugin_path = os.path.join(vlc_path, 'plugins')
                self.instance = vlc.Instance(['--no-xlib', '--plugin-path=' + plugin_path, '--vout=win32',  '--no-video-title-show'])
            else:
                self.instance = vlc.Instance(['--no-xlib'])

            self.player = self.instance.media_player_new()
            self.player.audio_set_volume(self.volume)
            print("VLC initialized successfully")

        except Exception as e:
            print(f"Error initializing VLC: {e}")
            sys.exit(1)

        # Initialize other components
        self.emergency = EmergencyHandler()
        self.accessibility = AccessibilityMode()
        
        # Tracks
        self.tracks = []
        self.current_track = 0
        self.supported_formats = ('.mp3', '.mp4', '.avi', '.mkv', '.wav')

        # Gesture handling
        self.prev_hand_y = None
        self.prev_hand_x = None
        self.gesture_cooldown = 0

        # Initialize web interface
        self.web_interface = WebInterface(self)

    # Rest of the methods remain unchanged
    def load_tracks(self, folder_path):
        """Load media files from specified folder"""
        try:
            self.tracks = []
            # Use absolute path and create folders if they don't exist
            folder = Path(folder_path).resolve()
            audio_folder = folder / "audio"
            video_folder = folder / "video"
            
            # Create folders if they don't exist
            audio_folder.mkdir(parents=True, exist_ok=True)
            video_folder.mkdir(parents=True, exist_ok=True)

            # Load from both audio and video folders
            for ext in self.supported_formats:
                self.tracks.extend(list(audio_folder.glob(f'*{ext}')))
                self.tracks.extend(list(video_folder.glob(f'*{ext}')))
            
            if self.tracks:
                print(f"Loaded {len(self.tracks)} media files")
                self.load_current_track()
                if self.is_playing:
                    self.player.play()
            else:
                print(f"No media files found in {folder}")
                print("Please add media files to:")
                print(f"- Audio folder: {audio_folder}")
                print(f"- Video folder: {video_folder}")
                print("Supported formats:", ", ".join(self.supported_formats))
                
        except Exception as e:
            print(f"Error loading tracks: {e}")

    def load_current_track(self):
        """Load current track with video support"""
        try:
            if not self.tracks:
                return
            
            if 0 <= self.current_track < len(self.tracks):
                track_path = str(self.tracks[self.current_track])
                print(f"Loading track: {track_path}")
            
                # Stop current playback and reset
                self.player.stop()
                self.player.set_media(None)
                
                # Create new media
                media = self.instance.media_new(track_path)
                self.player.set_media(media)
                
                # Check if current file is video
                self.is_video = self.is_video_file(track_path)
                print(f"Media type: {'Video' if self.is_video else 'Audio'}")
                
                # Set up video window if needed
                if self.is_video:
                    # Close existing window safely
                    try:
                        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                            cv2.destroyWindow(self.window_name)
                    except:
                        pass
                    
                    # Create new window
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_name, 800, 600)
                    
                    # Set window handle for Windows
                    if sys.platform.startswith('win32'):
                        try:
                            # Get window handle using Win32 API
                            hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
                            if hwnd:
                                self.player.set_hwnd(hwnd)
                                print("Video window created and attached")
                        except Exception as e:
                            print(f"Could not attach video window: {e}")
            
                # Setup video window if needed
                if self.is_video:
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_name, 800, 600)
                    if sys.platform.startswith('win32'):
                        hwnd = cv2.getWindowHandle(self.window_name)
                        if hwnd:
                            self.player.set_hwnd(hwnd)
                            print("Video window created")
            
                # Start playing if already in playing state
                if self.is_playing:
                    self.player.play()
            
        except Exception as e:
            print(f"Error loading track: {e}")

    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        try:
            return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return 0

    def detect_gestures(self, hand_landmarks):
        """Enhanced gesture detection with emergency support"""
        try:
            # Get landmarks
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Calculate distances
            thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
            fingers_distance = self.calculate_finger_distances(hand_landmarks)
            
            # Check if hand is raised (emergency gesture)
            hand_raised = wrist.y > thumb_tip.y and wrist.y > index_tip.y

            current_time = time.time()
            if current_time - self.gesture_cooldown < self.accessibility.cooldown_time:
                return

            # Check for emergency gesture (spread fingers and raised hand)
            if fingers_distance > 0.5 and hand_raised:  # Emergency threshold
                if not self.emergency.active:
                    self.emergency.trigger_emergency(self)
                    self.is_playing = False
                    print("Emergency stop triggered")
                    self.gesture_cooldown = current_time
                return

            # Handle regular gestures
            if self.accessibility.single_finger_mode:
                self.handle_single_finger_gestures(index_tip)
            else:
                self.handle_standard_gestures(thumb_index_distance, fingers_distance, index_tip, hand_landmarks)

            self.prev_hand_y = index_tip.y
            self.prev_hand_x = index_tip.x

        except Exception as e:
            print(f"Error in gesture detection: {e}")

    def calculate_finger_distances(self, hand_landmarks):
        """Calculate average finger distance from wrist"""
        try:
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            fingers = [
                self.mp_hands.HandLandmark.THUMB_TIP,
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP
            ]
            
            distances = [self.calculate_distance(hand_landmarks.landmark[finger], wrist) 
                        for finger in fingers]
            return sum(distances) / len(distances)
            
        except Exception as e:
            print(f"Error calculating finger distances: {e}")
            return 0

    def handle_standard_gestures(self, thumb_index_distance, fingers_distance, index_tip,hand_landmarks):
        """Handle standard gesture controls"""
        try:
            current_time = time.time()
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

            # Mute/Unmute - Make a fist
            if fingers_distance < 0.1 and not self.is_muted:
                self.player.audio_set_volume(0)
                self.is_muted = True
                self.gesture_cooldown = current_time
            elif fingers_distance > 0.3 and self.is_muted:
                self.player.audio_set_volume(self.volume)
                self.is_muted = False
                self.gesture_cooldown = current_time

            # Volume control - Hand up/down
            if self.prev_hand_y is not None and not self.is_muted:
                y_diff = index_tip.y - self.prev_hand_y
                if abs(y_diff) > 0.02:
                    self.volume = max(0, min(100, self.volume - int(y_diff * 100)))
                    self.player.audio_set_volume(self.volume)
                    self.gesture_cooldown = current_time

             # Track navigation - Hand left/right
            if self.prev_hand_x is not None:
                x_diff = index_tip.x - self.prev_hand_x
                if x_diff > 0.1:
                    self.next_track()
                    self.gesture_cooldown = current_time
                elif x_diff < -0.1:
                    self.previous_track()
                    self.gesture_cooldown = current_time

            # Play/Pause - Thumb and index finger pinch
            if thumb_index_distance < 0.1:
                self.toggle_play()
                self.gesture_cooldown = current_time

            # Switch between audio/video - Peace sign (index and middle finger up)
            # Check if only index and middle fingers are extended
            index_up = index_tip.y < wrist.y
            middle_up = middle_tip.y < wrist.y
            ring_down = ring_tip.y > wrist.y
            pinky_down = pinky_tip.y > wrist.y
            thumb_down = thumb_tip.y > index_tip.y
            
            if (index_up and middle_up and ring_down and pinky_down and thumb_down):
                if current_time - self.gesture_cooldown > 2.0:  # Cooldown to prevent rapid switching
                    current_media_type = 'video' if self.is_video else 'audio'
                    filtered_tracks = self.filter_media_type('video' if current_media_type == 'audio' else 'audio')
                
                    if filtered_tracks:
                        self.tracks = filtered_tracks
                        self.current_track = 0
                        self.load_current_track()
                        if self.is_playing:
                            self.player.play()
                        self.gesture_cooldown = current_time
                        new_mode = 'audio' if current_media_type == 'video' else 'video'
                        print(f"Switched to {new_mode.upper()} mode")
                
        except Exception as e:
            print(f"Error in standard gesture handling: {e}")

    def draw_status(self, frame):
        """Draw status information on frame"""
        try:
            # Draw volume status
            volume_status = "MUTED" if self.is_muted else f"Volume: {self.volume}%"
            cv2.putText(frame, volume_status, 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            # Draw play status
            play_status = "PAUSED" if not self.is_playing else "PLAYING"
            cv2.putText(frame, play_status, 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            # Draw media type and current/total tracks
            if self.tracks:
                media_type = "VIDEO" if self.is_video else "AUDIO"
                total_tracks = len(self.filter_media_type('video' if self.is_video else 'audio'))
                cv2.putText(frame, f"{media_type} Track: {self.current_track + 1}/{total_tracks}", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
                # Show current track name
                track_name = self.tracks[self.current_track].stem  # Using stem instead of name to remove extension
                cv2.putText(frame, f"Now Playing: {track_name}", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
            # Draw mode
            mode = "Accessibility Mode" if self.accessibility.single_finger_mode else "Standard Mode"
            cv2.putText(frame, mode, 
                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
        except Exception as e:
            print(f"Error drawing status: {e}")

    def handle_single_finger_gestures(self, index_tip):
        """Handle simplified gestures for accessibility mode"""
        try:
            current_time = time.time()
            
            # Play/Pause - Hold finger still in center area
            if self.prev_hand_y is not None and self.prev_hand_x is not None:
                # Check if finger is relatively still in the center
                y_diff = abs(index_tip.y - self.prev_hand_y)
                x_diff = abs(index_tip.x - self.prev_hand_x)
                
                # If finger is in center region (0.3-0.7 range) and still
                in_center_y = 0.3 < index_tip.y < 0.7
                in_center_x = 0.3 < index_tip.x < 0.7
                is_still = y_diff < 0.02 and x_diff < 0.02
                
                if in_center_y and in_center_x and is_still:
                    if current_time - self.gesture_cooldown > self.accessibility.cooldown_time:
                        self.toggle_play()
                        self.gesture_cooldown = current_time
                        print("Play/Pause toggled")
                        return
            
            if self.prev_hand_y is not None:
                # Vertical movement for volume
                y_diff = index_tip.y - self.prev_hand_y
                if abs(y_diff) > 0.04:
                    self.volume = max(0, min(100, self.volume - int(y_diff * 50)))
                    self.player.audio_set_volume(self.volume)
                    self.gesture_cooldown = current_time

            if self.prev_hand_x is not None:
                # Horizontal movement for track control
                x_diff = index_tip.x - self.prev_hand_x
                if abs(x_diff) > 0.15:
                    if x_diff > 0:
                        self.next_track()
                    else:
                        self.previous_track()
                    self.gesture_cooldown = current_time
                    
        except Exception as e:
            print(f"Error in accessibility gesture handling: {e}")

    def next_track(self):
        """Play next track"""
        try:
            if self.tracks:
                self.current_track = (self.current_track + 1) % len(self.tracks)
                self.load_current_track()
                if self.is_playing:
                    self.player.play()
        except Exception as e:
            print(f"Error changing to next track: {e}")

    def previous_track(self):
        """Play previous track"""
        try:
            if self.tracks:
                self.current_track = (self.current_track - 1) % len(self.tracks)
                self.load_current_track()
                if self.is_playing:
                    self.player.play()
        except Exception as e:
            print(f"Error changing to previous track: {e}")

    def toggle_play(self):
        """Toggle play/pause state"""
        try:
            if self.is_playing:
                self.player.pause()
                self.is_playing = False
            else:
                self.player.play()
                self.is_playing = True
        except Exception as e:
            print(f"Error toggling play state: {e}")
    
    def filter_media_type(self, media_type='all'):
        """Filter tracks by media type"""
        try:
            if media_type == 'video':
                return [t for t in self.tracks if self.is_video_file(str(t))]
            elif media_type == 'audio':
                return [t for t in self.tracks if not self.is_video_file(str(t))]
            return self.tracks
        except Exception as e:
            print(f"Error filtering media: {e}")
            return self.tracks

    def run(self):
        """Main application loop"""
        try:
            # Start web interface
            self.web_interface.start()
            print("Web interface running at http://localhost:5000")

            # Load media from current directory
            current_dir = Path(__file__).parent.parent
            media_path = current_dir / "media_folder"
            print(f"Loading media from: {media_path}")
            self.load_tracks(str(media_path))
        
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
             raise Exception("Could not open camera")

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

                # Handle video window if playing video file
                if self.is_video:
                    try:
                        # Check if window exists
                        window_exists = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0
                        
                        if not window_exists:
                            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(self.window_name, 800, 600)
                            
                            # Set window handle for Windows
                            if sys.platform.startswith('win32'):
                                try:
                                    time.sleep(0.1)  # Give window time to create
                                    hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
                                    if hwnd:
                                        self.player.set_hwnd(hwnd)
                                except Exception as e:
                                    print(f"Could not attach video window: {e}")
                        
                        # Update video window title
                        if self.tracks and window_exists:
                            track_name = self.tracks[self.current_track].stem
                            cv2.setWindowTitle(self.window_name, f"Playing: {track_name}")
                    except Exception as e:
                        print(f"Error handling video window: {e}")

                # Draw status
                self.draw_status(frame)
                self.draw_gesture_guide(frame)
            
                # Show emergency status
                if self.emergency.active:
                    cv2.putText(frame, "EMERGENCY STOP ACTIVE", 
                              (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 0, 255), 2)
            
                cv2.imshow('Gesture Control', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 27:  # ESC key for emergency
                    if self.emergency.active:
                        self.emergency.reset()
                        print("Emergency reset")
                    else:
                        self.emergency.trigger_emergency(self)
                        self.is_playing = False
                elif key == ord('m'):  # Toggle accessibility mode
                    self.accessibility.single_finger_mode = not self.accessibility.single_finger_mode
                    print(f"Accessibility mode: {'ON' if self.accessibility.single_finger_mode else 'OFF'}")
                elif key == ord('p'):  # Play/Pause 
                    self.toggle_play()
                elif key == ord('n'):  # Next track
                    self.next_track()
                elif key == ord('b'):  # Previous track 
                    self.previous_track()
                elif key == ord('r'):  # Reset emergency
                    if self.emergency.active:
                        self.emergency.reset()
                        self.player.audio_set_volume(self.volume)
                        print("Emergency reset - playback can resume")

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if 'cap' in locals():
                cap.release()
            
            # Safely close all windows
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
            # Close MediaPipe hands
            try:
                self.hands.close()
            except:
                pass
            
            # Stop playback when closing
            try:
                if self.player:
                    self.player.stop()
            except:
                pass
            
            # Stop web interface
            try:
                self.web_interface.stop()
            except:
                pass

    def is_video_file(self, file_path):
        """Check if file is a video"""
        return Path(file_path).suffix.lower() in ['.mp4', '.avi', '.mkv']

    
    def draw_gesture_guide(self, frame):
        """Draw gesture guide on frame"""
        try:
            if self.accessibility.single_finger_mode:
                # Accessibility mode gestures
                guides = [
                    "ACCESSIBILITY MODE:",
                    "Vertical Finger: Volume Up/Down",
                    "Horizontal Finger: Previous/Next",
                    "Hold Center: Play/Pause",
                    "Keys: M-Mode, ESC/R-Reset"
                ]
            else:
                # Standard mode gestures
                guides = [
                    "STANDARD MODE:",
                    "Vertical Hand: Volume Up/Down",
                    "Horizontal Hand: Previous/Next",
                    "Pinch: Play/Pause",
                    "Peace Sign: Switch Audio/Video",
                    "Spread Fingers Up: Emergency Stop",
                    "Fist: Mute/Unmute",
                    "Keys: M-Mode, P-Play, ESC/R-Reset"
                ]
        
            y_pos = 250
            for guide in guides:
                cv2.putText(frame, guide, 
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)
                y_pos += 30
            
        except Exception as e:
            print(f"Error drawing gesture guide: {e}")

if __name__ == "__main__":
    try:
        player = GestureMediaPlayer()
        player.run()
    except Exception as e:
        print(f"Application error: {e}")