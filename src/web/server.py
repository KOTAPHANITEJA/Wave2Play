from flask import Flask, render_template, jsonify, request
import threading

class WebInterface:
    def __init__(self, player):
        self.app = Flask(__name__)
        self.player = player
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/')
        def home():
            return render_template('index.html')
            
        @self.app.route('/api/status')
        def status():
            current_track_index = self.player.current_track
            total_tracks = len(self.player.tracks)
            
            return jsonify({
                'playing': self.player.is_playing,
                'volume': self.player.volume,
                'muted': self.player.is_muted,
                'track': str(self.player.tracks[current_track_index].stem) if self.player.tracks else "",
                'currentTrack': current_track_index + 1,
                'totalTracks': total_tracks,
                'mode': "accessibility" if self.player.accessibility.single_finger_mode else "standard",
                'emergency': self.player.emergency.active,
                'isVideo': self.player.is_video
            })
        
        @self.app.route('/api/control/<action>', methods=['POST'])
        def control(action):
            try:
                if action == 'play':
                    self.player.toggle_play()
                elif action == 'next':
                    self.player.next_track()
                elif action == 'previous':
                    self.player.previous_track()
                elif action == 'mode':
                    self.player.accessibility.single_finger_mode = not self.player.accessibility.single_finger_mode
                elif action == 'emergency':
                    if self.player.emergency.active:
                        self.player.emergency.reset()
                        self.player.player.audio_set_volume(self.player.volume)
                    else:
                        self.player.emergency.trigger_emergency(self.player)
                        self.player.is_playing = False
                elif action == 'volume':
                    data = request.get_json()
                    volume = int(data.get('volume', 50))
                    self.player.volume = max(0, min(100, volume))
                    self.player.player.audio_set_volume(self.player.volume)
                    
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
            
    def start(self):
        """Start the web server"""
        threading.Thread(target=lambda: self.app.run(port=5000, debug=False),
                       daemon=True).start()
    
    def stop(self):
        """Stop the web server"""
        pass