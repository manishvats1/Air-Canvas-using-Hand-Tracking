import cv2
import numpy as np
import mediapipe as mp
import os
import time
import tkinter as tk
from tkinter import filedialog

brush_size = 5
brush_type = 'line'
color_index = 0
last_point = None
base_image = None
point_history = []
shape_mode = False
selected_shape = None
shape_drag = False
shape_resize = False
active_shape = None
shapes_on_canvas = []  
current_text = ""
text_position = None

mouse_dragging = False
mouse_resizing = False
dragged_item = None  
drag_offset = (0, 0)
text_items = []  
last_mouse_pos = None

SIDEBAR_WIDTH = 73  
CANVAS_WIDTH = 636
WINDOW_HEIGHT = 471
WINDOW_WIDTH = CANVAS_WIDTH + SIDEBAR_WIDTH

def mouse_callback(event, x, y, flags, param):
    global mouse_dragging, mouse_resizing, dragged_item, drag_offset, last_mouse_pos
    
    if x <= SIDEBAR_WIDTH:  
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        
        for shape in shapes_on_canvas:
            shape_x, shape_y = shape['pos']
            if abs(x - shape_x) < 20 and abs(y - shape_y) < 20:
                mouse_dragging = True
                dragged_item = shape
                drag_offset = (x - shape_x, y - shape_y)
                return

        for text in text_items:
            text_x, text_y = text['pos']
            if abs(x - text_x) < 50 and abs(y - text_y) < 20:
                mouse_dragging = True
                dragged_item = text
                drag_offset = (x - text_x, y - text_y)
                return

    elif event == cv2.EVENT_RBUTTONDOWN:
        for shape in shapes_on_canvas:
            shape_x, shape_y = shape['pos']
            if abs(x - shape_x) < 20 and abs(y - shape_y) < 20:
                mouse_resizing = True
                dragged_item = shape
                last_mouse_pos = (x, y)
                return

        for text in text_items:
            text_x, text_y = text['pos']
            if abs(x - text_x) < 50 and abs(y - text_y) < 20:
                mouse_resizing = True
                dragged_item = text
                last_mouse_pos = (x, y)
                return

    elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
        mouse_dragging = False
        mouse_resizing = False
        dragged_item = None
        last_mouse_pos = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

cv2.namedWindow('Paint')
cv2.setMouseCallback('Paint', mouse_callback)
mp_draw = mp.solutions.drawing_utils

tk.Tk().withdraw()

colors = [
    (0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0),
    (255, 0, 0), (0, 255, 255), (128, 0, 128), (255, 191, 0),
    (0, 165, 255), (147, 20, 255), (0, 215, 255), (128, 128, 128)
]
color_labels = ['Black', 'White', 'Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Cyan',
              'Orange', 'Pink', 'Gold', 'Gray']
brush_sizes = [2, 4, 6, 8]
brush_types = ['Natural', 'Calligraphy', 'Airbrush', 'Oil', 'Crayon', 'Pencil']
brush_type = 'Natural'


shapes = ['Circle', 'Rectangle', 'Square', 'Triangle', 'Pentagon', 
          'Hexagon', 'Star', 'Arrow', 'Diamond', 'Heart']
shape_size = 50  

paintWindow = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255
paint_image = np.ones((WINDOW_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255

def smooth_point(curr, alpha=0.9, buffer_size=5):
    global point_history
    point_history.append(curr)
    if len(point_history) > buffer_size:
        point_history.pop(0)
    if len(point_history) < 2:
        return curr
    x = int(np.mean([p[0] for p in point_history]))
    y = int(np.mean([p[1] for p in point_history]))
    return (x, y)

def save_drawing():
    root = tk.Tk()
    root.withdraw()

    default_name = f'drawing_{int(time.time())}.png'
    
    file_path = filedialog.asksaveasfilename(
        defaultextension='.png',
        initialfile=default_name,
        filetypes=[
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg;*.jpeg'),
            ('All files', '*.*')
        ],
        title='Save your drawing'
    )
    
    if file_path:
        try:
            img_to_save = cv2.cvtColor(paint_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(file_path, img_to_save)
            print(f'Drawing saved successfully as: {file_path}')
        except Exception as e:
            print(f'Error saving file: {str(e)}')
    
    root.destroy()

class VirtualKeyboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Virtual Keyboard")
        self.root.geometry("600x300")
        self.text = ""
        self.create_widgets()

    def create_widgets(self):
        self.text_display = tk.Text(self.root, height=3, width=50)
        self.text_display.pack(pady=10)

        keys = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '?'],
            ['Space', 'Backspace', 'Done']
        ]

        for row in keys:
            frame = tk.Frame(self.root)
            frame.pack(pady=2)
            for key in row:
                if key == 'Space':
                    btn = tk.Button(frame, text=key, width=20, command=lambda: self.press_key(' '))
                elif key == 'Backspace':
                    btn = tk.Button(frame, text=key, width=10, command=self.backspace)
                elif key == 'Done':
                    btn = tk.Button(frame, text=key, width=10, command=self.done)
                else:
                    btn = tk.Button(frame, text=key, width=4, command=lambda k=key: self.press_key(k))
                btn.pack(side=tk.LEFT, padx=2)

    def press_key(self, key):
        global current_text
        current_text += key
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, current_text)

    def backspace(self):
        global current_text
        current_text = current_text[:-1]
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, current_text)

    def done(self):
        global text_mode, text_position, text_items
        if current_text:
            pos = text_position if text_position else (SIDEBAR_WIDTH + CANVAS_WIDTH//2, WINDOW_HEIGHT//2)
            text_items.append({
                'text': current_text,
                'pos': pos,
                'size': 1.0,  
                'color': colors[color_index]
            })

def draw_sidebar(img):
    button_height = 50  
    spacing = 12 
    button_margin = 12 
    y_pos = 10
    labels = ["CLR", "IMG" if base_image is None else "RMV", "SIZE", "BRUSH", "SHAPE", "TEXT", "SAVE"]
    for i, label in enumerate(labels):
        if (size_mode and i != 2) or (brush_mode and i != 3) or (shape_mode and i != 4):
            color = (40, 40, 40)
        else:
            color = (60, 60, 60)
        cv2.rectangle(img, (button_margin, y_pos), (SIDEBAR_WIDTH - button_margin, y_pos + button_height), color, -1)
        cv2.putText(img, label, (button_margin + 6, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += button_height + spacing

def draw_color_palette(img):
    for i, color in enumerate(colors):
        x1 = SIDEBAR_WIDTH + 10 + i * 40
        x2 = x1 + 30
        cv2.rectangle(img, (x1, 10), (x2, 40), color, -1)
        if i == color_index:
            cv2.rectangle(img, (x1-2, 8), (x2+2, 42), (0, 0, 0), 2)

        label = color_labels[i]
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        text_x = x1 + (30 - text_size[0]) // 2
        cv2.putText(img, label, (text_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    exit_btn_x1 = CANVAS_WIDTH - 70 
    exit_btn_x2 = CANVAS_WIDTH - 10
    exit_btn_y1 = 65 
    exit_btn_y2 = 95
    
    cv2.rectangle(img, (exit_btn_x1, exit_btn_y1), (exit_btn_x2, exit_btn_y2), (0, 0, 200), -1)
    
    text = "EXIT"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    text_x = exit_btn_x1 + (60 - text_size[0]) // 2
    text_y = exit_btn_y1 + (30 + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_shape(img, center, color, shape_name, size=None):
    x, y = center
    if size is None:
        size = shape_size
    
    if shape_name == 'Circle':
        cv2.circle(img, center, size, color, -1)
    
    elif shape_name == 'Rectangle':
        cv2.rectangle(img, (x-size, y-size//2), (x+size, y+size//2), color, -1)
    
    elif shape_name == 'Square':
        cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color, -1)
    
    elif shape_name == 'Triangle':
        pts = np.array([[x, y-size], [x-size, y+size], [x+size, y+size]], np.int32)
        cv2.fillPoly(img, [pts], color)
    
    elif shape_name == 'Pentagon':
        pts = []
        for i in range(5):
            angle = i * 2 * np.pi / 5 - np.pi / 2
            pt_x = x + size * np.cos(angle)
            pt_y = y + size * np.sin(angle)
            pts.append([int(pt_x), int(pt_y)])
        cv2.fillPoly(img, [np.array(pts)], color)
    
    elif shape_name == 'Hexagon':
        pts = []
        for i in range(6):
            angle = i * 2 * np.pi / 6
            pt_x = x + size * np.cos(angle)
            pt_y = y + size * np.sin(angle)
            pts.append([int(pt_x), int(pt_y)])
        cv2.fillPoly(img, [np.array(pts)], color)
    
    elif shape_name == 'Star':
        pts = []
        for i in range(10):
            angle = i * 2 * np.pi / 10 - np.pi / 2
            r = size if i % 2 == 0 else size//2
            pt_x = x + r * np.cos(angle)
            pt_y = y + r * np.sin(angle)
            pts.append([int(pt_x), int(pt_y)])
        cv2.fillPoly(img, [np.array(pts)], color)
    
    elif shape_name == 'Arrow':
        cv2.rectangle(img, (x-size//4, y-size//8), (x+size//2, y+size//8), color, -1)
        pts = np.array([[x+size//2, y], [x+size, y], [x+size//2, y-size//2]], np.int32)
        cv2.fillPoly(img, [pts], color)
    
    elif shape_name == 'Diamond':
        pts = np.array([[x, y-size], [x+size, y], [x, y+size], [x-size, y]], np.int32)
        cv2.fillPoly(img, [pts], color)
    
    elif shape_name == 'Heart':
        radius = size // 4
        cv2.circle(img, (x-radius, y-radius), radius, color, -1)
        cv2.circle(img, (x+radius, y-radius), radius, color, -1)
        pts = np.array([[x-size//2, y-radius], [x+size//2, y-radius], [x, y+size//2]], np.int32)
        cv2.fillPoly(img, [pts], color)

def apply_brush_effect(img, start_point, end_point, color, size, brush_type):
    if brush_type == 'Natural':
        cv2.line(img, start_point, end_point, color, size)
    
    elif brush_type == 'Calligraphy':
        angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        thickness = int(size * abs(np.sin(angle)) + size/2)
        cv2.line(img, start_point, end_point, color, thickness)
    
    elif brush_type == 'Airbrush':
        cv2.line(img, start_point, end_point, color, 1)
        for _ in range(size * 2):
            x = np.random.randint(min(start_point[0], end_point[0]) - size, max(start_point[0], end_point[0]) + size)
            y = np.random.randint(min(start_point[1], end_point[1]) - size, max(start_point[1], end_point[1]) + size)
            cv2.circle(img, (x, y), 1, color, -1)
    
    elif brush_type == 'Oil':
        for i in range(3):
            offset = np.random.randint(-size//2, size//2, 2)
            pt1 = (start_point[0] + offset[0], start_point[1] + offset[1])
            pt2 = (end_point[0] + offset[0], end_point[1] + offset[1])
            cv2.line(img, pt1, pt2, color, size//2)
    
    elif brush_type == 'Crayon':
        for i in range(size):
            offset = np.random.randint(-2, 3, 2)
            pt1 = (start_point[0] + offset[0], start_point[1] + offset[1])
            pt2 = (end_point[0] + offset[0], end_point[1] + offset[1])
            cv2.line(img, pt1, pt2, color, 1)
    
    elif brush_type == 'Pencil':
        cv2.line(img, start_point, end_point, color, 1)
        for i in range(size//2):
            offset = np.random.randint(-1, 2, 2)
            pt1 = (start_point[0] + offset[0], start_point[1] + offset[1])
            pt2 = (end_point[0] + offset[0], end_point[1] + offset[1])
            cv2.line(img, pt1, pt2, color, 1)

def draw_shape_palette(img):
    if shape_mode or selected_shape:
        if selected_shape:
            msg = f"Selected: {selected_shape}"
            msg2 = "Join fingers to place shape!"
        else:
            msg = "Select shape!"
            msg2 = ""
        
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = SIDEBAR_WIDTH + 10
        cv2.putText(img, msg, (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if msg2:
            cv2.putText(img, msg2, (text_x, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for i, shape in enumerate(shapes):
            y1 = 100 + i * 35  
            x1 = SIDEBAR_WIDTH + 10
            x2 = x1 + 150  
            y2 = y1 + 30

def process_click(x, y):
    global size_mode, brush_mode, brush_type, brush_size, base_image, shape_mode, selected_shape
    global text_mode, keyboard_window, current_text, text_position, color_picker_mode

def draw_brush_palette(img):
    if brush_mode:
        msg = "Select brush style!"
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = SIDEBAR_WIDTH + (CANVAS_WIDTH - text_size[0]) // 2
        cv2.putText(img, msg, (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for i, brush in enumerate(brush_types):
            x1 = SIDEBAR_WIDTH + 10 + i * 100
            x2 = x1 + 90
            if brush_type == brush:
                cv2.rectangle(img, (x1, 100), (x2, 130), (200, 200, 200), -1)
            cv2.putText(img, brush, (x1 + 5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def draw_size_palette(img):
    if size_mode:
        msg = "Select size first!"
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = SIDEBAR_WIDTH + (CANVAS_WIDTH - text_size[0]) // 2
        cv2.putText(img, msg, (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for i, size in enumerate(brush_sizes):
            x1 = SIDEBAR_WIDTH + 10 + i * 60
            x2 = x1 + 50
            center = (x1 + 25, 110)
            cv2.circle(img, center, size, (0, 0, 0), -1)
            if brush_size == size:
                cv2.circle(img, center, size + 2, (0, 255, 0), 1)  
            text_size = cv2.getTextSize(str(size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 + (50 - text_size[0]) // 2
            cv2.putText(img, str(size), (text_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def handle_image_button():
    global base_image, paint_image
    if base_image is None:
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:

                img = cv2.resize(img, (CANVAS_WIDTH, WINDOW_HEIGHT))
                base_image = img.copy()
                paint_image = img.copy()
    else:
        base_image = None
        paint_image = np.ones((WINDOW_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255


size_mode = False
brush_mode = False
color_picker_mode = False
selection_required = False  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CANVAS_WIDTH, WINDOW_HEIGHT))
    h, w, _ = frame.shape
    output = np.ones((WINDOW_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255

    if base_image is not None:
        output[:, :] = base_image
    else:
        output[67:, :] = frame[67:, :]

    sidebar = np.ones((WINDOW_HEIGHT, SIDEBAR_WIDTH, 3), dtype=np.uint8) * 40
    draw_sidebar(sidebar)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_color_palette(output)
    draw_size_palette(output)
    draw_brush_palette(output)
    draw_shape_palette(output)

    paintWindow = paint_image.copy()
    
    for shape in shapes_on_canvas:
        draw_shape(paintWindow, shape['pos'], shape['color'], shape['name'], shape['size'])
    
    for text_item in text_items:
        cv2.putText(paintWindow, text_item['text'], text_item['pos'],
                    cv2.FONT_HERSHEY_SIMPLEX, text_item['size'], 
                    text_item['color'], 2)

    output[:, :SIDEBAR_WIDTH] = sidebar

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        ix, iy = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)
        tx, ty = int(lm.landmark[4].x * w), int(lm.landmark[4].y * h)
        center = (ix, iy)
        distance = np.linalg.norm(np.array([ix, iy]) - np.array([tx, ty]))
        drawing = distance > 30  

        for id, landmark in enumerate(lm.landmark):
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            if id == 8:  
                cv2.circle(output, (px, py), 8, (0, 255, 0), -1) 
            else:
                cv2.circle(output, (px, py), 3, (255, 0, 0), -1)  
        mp_draw.draw_landmarks(output, lm, mp_hands.HAND_CONNECTIONS)

        if 12 <= ix <= SIDEBAR_WIDTH - 12:
            y = iy
            if 10 <= y <= 60: 
                if base_image is not None:
                    paint_image = base_image.copy()
                else:
                    paint_image[:, :] = 255
                shapes_on_canvas.clear()
                last_point = None
            elif 80 <= y <= 130:  
                handle_image_button()
                last_point = None

            elif 150 <= y <= 200: 
                size_mode = not size_mode
                brush_mode = False
                shape_mode = False
                text_mode = False
                if size_mode:
                    selection_required = True 
            elif 220 <= y <= 270:  
                brush_mode = not brush_mode
                size_mode = False
                shape_mode = False
                text_mode = False
                if brush_mode:
                    selection_required = True 
            elif 285 <= y <= 335:  
                shape_mode = not shape_mode
                size_mode = False
                brush_mode = False
                text_mode = False
                if shape_mode:
                    selection_required = True 
                else:
                    selected_shape = None 
            elif 350 <= y <= 400: 
                text_mode = not text_mode
                shape_mode = False
                size_mode = False
                brush_mode = False
                if text_mode:  
                    current_text = ""
                    keyboard_window = VirtualKeyboard()
                    keyboard_window.root.mainloop()
            elif 415 <= y <= 465:  
                save_drawing()

        if 10 <= iy <= 55 and ix > SIDEBAR_WIDTH:
            rel_x = ix - SIDEBAR_WIDTH - 10
            idx = rel_x // 40
            if 0 <= idx < len(colors):
                color_index = idx

        if 65 <= iy <= 95 and CANVAS_WIDTH - 70 <= ix <= CANVAS_WIDTH - 10:
            cv2.destroyAllWindows()
            cap.release()
            exit()

        if size_mode and 77 <= iy <= 121 and ix > SIDEBAR_WIDTH:
            rel_x = ix - SIDEBAR_WIDTH - 10
            idx = rel_x // 60
            if 0 <= idx < len(brush_sizes):
                brush_size = brush_sizes[idx]
                size_mode = False  
                selection_required = False  
                last_point = None  

        if brush_mode and 110 <= iy <= 143 and ix > SIDEBAR_WIDTH:
            rel_x = ix - SIDEBAR_WIDTH - 10
            idx = rel_x // 100
            if 0 <= idx < len(brush_types):
                brush_type = brush_types[idx]
                brush_mode = False
                selection_required = False  
                last_point = None  

        if shape_mode and SIDEBAR_WIDTH + 12 <= ix <= SIDEBAR_WIDTH + 176:
            idx = (iy - 100) // 35  
            if 0 <= idx < len(shapes):
                selected_shape = shapes[idx]
                selection_required = False  
                last_point = None  
        
        if selected_shape and ix > SIDEBAR_WIDTH + 160:  
            if last_finger_state and not drawing:  
                shapes_on_canvas.append({
                    'name': selected_shape,
                    'pos': (ix, iy),
                    'size': shape_size,
                    'color': colors[color_index]
                })
                shape_mode = False
                selected_shape = None
                selection_required = False 
            
            for shape in shapes_on_canvas:
                shape_x, shape_y = shape['pos']
                if abs(ix - shape_x) < 20 and abs(iy - shape_y) < 20:
                    if drawing:  
                        active_shape = shape
                        shape_drag = True
                        shape['pos'] = (ix, iy)
                    else:  
                        shape_resize = True
                        active_shape = shape
                        size_change = (last_point[1] - iy) if last_point else 0
                        shape['size'] = max(10, shape['size'] + size_change)
            
            if not drawing:
                shape_drag = False
                active_shape = None
            
            temp_image = paint_image.copy()


        if drawing and iy > 67 and not (selection_required or size_mode or brush_mode or shape_mode or text_mode):
            pt = smooth_point((ix, iy))
            if last_point is not None:
                apply_brush_effect(paint_image, last_point, pt, colors[color_index], brush_size, brush_type)
                apply_brush_effect(output, last_point, pt, colors[color_index], brush_size, brush_type)
            last_point = pt
        else:
            last_point = None
            point_history.clear()

    cv2.imshow("Output", output)
    cv2.imshow("Paint", paintWindow)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()