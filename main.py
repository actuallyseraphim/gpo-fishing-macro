import numpy as np
import cv2
import mss
import time
from collections import deque
import pynput

bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

def detect_blue(x):
    c = np.array([255, 170, 85, 0])
    i = cv2.inRange(x, c, c+1)
    y, x = np.where(i >= 0.99)
    return (y.min(), x.min(), y.max(), x.max())

def detect_gray(x):
    c = np.array([25, 25, 25, 0])
    i = cv2.inRange(x, c-1, c+1)
    y, x = np.where(i >= 0.99)
    return (y.min(), x.min(), y.max(), x.max())

def dark_gray_count(img, center=25, tol=10):
    lower = np.array([center - tol]*3+[0], dtype=np.uint8)
    upper = np.array([center + tol]*3+[0], dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    return cv2.countNonZero(mask)

def bar(x):
    c = detect_blue(x)
    dx = (c[3]-c[1])*2
    dy = (c[2]-c[0])//3
    i0 = x[c[0]-dx:c[2]+dy+dx, c[1]-dx:c[3]+dx]
    i1 = x[c[0]-dy-dx:c[2]+dx, c[1]-dx:c[3]+dx]
    dgpi0 = dark_gray_count(i0)
    dgpi1 = dark_gray_count(i1)
    if abs(dgpi0 - dgpi1) <= dx*dy/3:
        return x[c[0]:c[2], c[1]:c[3]]
    if dgpi0 >= dgpi1:
        return x[c[0]:c[2]+dy, c[1]:c[3]]
    else:
        return x[c[0]-dy:c[2], c[1]:c[3]]

def zone_line_pos(x):
    i = bar(x)[3:]
    av = i.mean(axis=1)
    b = av[:,0]
    g = av[:,1:].mean(axis=1)
    bm = np.where(b < 200)[0]
    z = (bm[0]+bm[-1])/(2*av.shape[0])
    wm = np.where(g > 100)[0]
    l = (wm[0]+wm[-1])/(2*av.shape[0])
    return (1-z, 1-l)

PREDICT_AHEAD = 0.25
ACCEL_MULTIPLIER = 1.75
BRAKE_GAIN = 1.25
VEL_SMOOTH_ALPHA = 0.25
ACC_SMOOTH_ALPHA = 0.25

zone_pos_buf = deque(maxlen=10)
start_time = np.inf

smoothed_vel = 0.0
smoothed_accel = 0.0
prev_raw_vel = None
prev_vel_time = None

def estimate_motion(t, p):
    global smoothed_vel, smoothed_accel, prev_raw_vel, prev_vel_time
    zone_pos_buf.append((t, p))    
    vel = smoothed_vel
    accel = smoothed_accel    
    if len(zone_pos_buf) >= 2:
        t0, p0 = zone_pos_buf[-2]
        t1, p1 = zone_pos_buf[-1]
        dt = max(1e-6, t1 - t0)
        raw_vel = (p1 - p0) / dt
        smoothed_vel = VEL_SMOOTH_ALPHA * raw_vel + (1 - VEL_SMOOTH_ALPHA) * smoothed_vel
        vel = smoothed_vel
        if prev_raw_vel is not None and prev_vel_time is not None:
            dtv = max(1e-6, t1 - prev_vel_time)
            raw_accel = (raw_vel - prev_raw_vel) / dtv
            smoothed_accel = ACC_SMOOTH_ALPHA * raw_accel + (1 - ACC_SMOOTH_ALPHA) * smoothed_accel
            accel = smoothed_accel
        prev_raw_vel = raw_vel
        prev_vel_time = t1
    return vel, accel

def predict_zone(p_now, vel, accel):
    pred = p_now + vel * PREDICT_AHEAD + 0.5 * accel * (PREDICT_AHEAD ** 2) * ACCEL_MULTIPLIER
    pred -= BRAKE_GAIN * accel * (PREDICT_AHEAD ** 2) * 0.5
    pred = max(0.0, min(1.0, pred))
    return pred

mouse = pynput.mouse.Controller()
keyboard = pynput.keyboard.Controller()
def controls(c): # for later crossplatfomatability
    if c == 'mouse_down':
        mouse.press(pynput.mouse.Button.left)
    if c == 'mouse_up':
        mouse.release(pynput.mouse.Button.left)
    if c == 'click':
        mouse.release(pynput.mouse.Button.left)
        mouse.press(pynput.mouse.Button.left)
        time.sleep(0.1)
        mouse.release(pynput.mouse.Button.left)
    if c == 'zone_up':
        controls('mouse_down')
    if c == 'zone_down':
        controls('mouse_up')
    if c == 'throw': # assumes the rod is in the first slot
        keyboard.press('1')
        keyboard.release('1')
        time.sleep(0.2)
        keyboard.press('1')
        keyboard.release('1')
        time.sleep(0.1)
        controls('click')        

while True:
    with mss.mss() as sct:
        img = np.asarray(sct.grab(bounding_box))
        print(img.shape)
    try:
        zone_pos, line_pos = zone_line_pos(img)
        print(zone_pos, line_pos)
        t = time.monotonic()

        zone_vel, zone_acc = estimate_motion(t, zone_pos)
        zone_pred = predict_zone(zone_pos, zone_vel, zone_acc)

        if (line_pos > zone_pred):
            controls('zone_up')
        elif (line_pos < zone_pred):
            controls('zone_down')

        start_time = None
        thrown = True

        print(f"zone_pos={zone_pos:.3f} zone_pred={zone_pred:.3f} line_pos={line_pos:.3f} zone_vel={zone_vel:.3f} zone_acc={zone_acc:.3f}")

    except Exception as e:
        print(e)
        if start_time is None:
            controls('mouse_up')
            start_time = time.monotonic()
        if (time.perf_counter() - start_time > 2):
            controls('throw')
            print('Thrown')
            start_time = np.inf

