from src.gesture_player import GestureMediaPlayer

if __name__ == "__main__":
    try:
        player = GestureMediaPlayer()
        player.run()
    except Exception as e:
        print(f"Application error: {e}")