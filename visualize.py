import numpy as np
import cv2

class FrameNode:
    def __init__(self, frame_index, landmarks):
        self.frame_index = frame_index
        self.landmarks = landmarks  # Array of shape (75, 2)
        self.prev = None
        self.next = None

class VideoLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.current = None
        self.size = 0

    def append(self, frame_index, landmarks):
        new_node = FrameNode(frame_index, landmarks)
        if not self.head:
            self.head = new_node
            self.tail = new_node
            self.current = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.size += 1

def render_frame(node, canvas_size=(600, 600)):
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    
    if node is None:
        return canvas

    landmarks = node.landmarks
    margin = 50 
    
    COLOR_POSE = (255, 200, 0)       # Cyan/Blue
    COLOR_LEFT_HAND = (0, 0, 255)    # Red
    COLOR_RIGHT_HAND = (0, 255, 255) # Yellow
    
    for i, (x, y) in enumerate(landmarks):
        # 1. Skip NaNs
        if np.isnan(x) or np.isnan(y):
            continue
            
        # 2. Skip imputed (0,0) points
        if abs(x) < 1e-5 and abs(y) < 1e-5:
            continue

        # 3. Skip points that extend outside the visible camera bounds (0.0 to 1.0)
        if not (0.0 < x < 1.0 and 0.0 < y < 1.0):
            continue

        # Color assignment
        if i < 33:
            color = COLOR_POSE
        elif i < 54:
            color = COLOR_LEFT_HAND
        else:
            color = COLOR_RIGHT_HAND

        # Scale coordinates to canvas
        draw_x = int(x * (canvas_size[0] - 2 * margin)) + margin
        draw_y = int(y * (canvas_size[1] - 2 * margin)) + margin
        
        cv2.circle(canvas, (draw_x, draw_y), 3, color, -1)

    cv2.putText(canvas, f"Frame: {node.frame_index}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return canvas

def main():
    # Update directory path as needed
    data_path = r'C:\Users\vanlo\VSC\ASL\data\processed\2044\31571477.npy'
    data = np.load(data_path)
    
    num_frames = data.shape[0]
    video_list = VideoLinkedList()
    
    for i in range(num_frames):
        frame_data = data[i].reshape(-1, 2)
        video_list.append(i, frame_data)

    video_list.current = video_list.head
    
    print("Controls:")
    print(" [d] or [Right Arrow] : Next Frame")
    print(" [a] or [Left Arrow]  : Previous Frame")
    print(" [v]                  : Play/Pause Video Mode")
    print(" [q]                  : Quit")

    video_mode = False

    while True:
        canvas = render_frame(video_list.current)
        cv2.imshow("Skeleton Viewer", canvas)
        
        delay = 30 if video_mode else 0
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('v'):
            video_mode = not video_mode
            
        elif not video_mode:
            if key == ord('d') or key == 83: # 'd' or Right Arrow
                if video_list.current.next:
                    video_list.current = video_list.current.next
            elif key == ord('a') or key == 81: # 'a' or Left Arrow
                if video_list.current.prev:
                    video_list.current = video_list.current.prev
                    
        elif video_mode:
            if video_list.current.next:
                video_list.current = video_list.current.next
            else:
                video_mode = False

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()