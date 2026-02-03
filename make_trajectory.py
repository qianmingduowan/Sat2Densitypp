import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from matplotlib.widgets import Cursor

import os



def parse_args():
    # Parse CLI arguments
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument('--input_img_path', type=str, required=True)
    parse_args.add_argument('--work_dir', type=str, required=True)
    parse_args.add_argument('--num_of_point', type=int, default=79)
    parse_args.add_argument('--force', action='store_true')
    return parse_args.parse_args()


def select_points(sat_image,num_of_point):
    fig = plt.figure()
    fig.set_size_inches(1,1,forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.imshow(sat_image)

    coords = []
    fig.add_axes(ax)

    def on_press(event):
        global dragging
        if event.button == 1 and event.inaxes == ax:  # left mouse button and inside axes
            dragging = True
            print(f"Start at ({float(event.xdata)}, {float(event.ydata)})")

    def on_drag(event):
        global dragging
        if not dragging or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        coords.append((x, y))

    def on_release(event):
        global dragging
        if event.button == 1 and dragging:
            dragging = False
            print("Mouse released. Recording stopped.")
            fig.canvas.mpl_disconnect(press_cid)
            fig.canvas.mpl_disconnect(drag_cid)
            fig.canvas.mpl_disconnect(release_cid)
            plt.close(fig)

    # connect the event handlers
    press_cid = fig.canvas.mpl_connect('button_press_event', on_press)
    drag_cid = fig.canvas.mpl_connect('motion_notify_event', on_drag)
    release_cid = fig.canvas.mpl_connect('button_release_event', on_release)
    plt.show()
    
    pixels = np.array(list(dict.fromkeys(coords))) # remove duplicates while preserving order
    tck, u = splprep(pixels.T, s=25, per=0)
    u_new = np.linspace(u.min(), u.max(), num_of_point+1)
    x_new, y_new = splev(u_new, tck)
    smooth_path = np.array([x_new,y_new]).T
    angles = np.arctan2(y_new[1:]-y_new[:-1],x_new[1:]-x_new[:-1])
    
    # final visualization
    fig_final, ax_final = plt.subplots()
    ax_final.imshow(sat_image)
    ax_final.plot(pixels[:,0], pixels[:,1], 'o', color='red')
    ax_final.plot(smooth_path[:,0], smooth_path[:,1], '-', color='blue')
    plt.show()

    return pixels, angles, smooth_path,fig_final


def main():
    args = parse_args()
    save_path = os.path.join(args.work_dir, os.path.basename(args.input_img_path).rsplit('.', 1)[0])
    os.makedirs(save_path, exist_ok=True)

    save_csv = os.path.join(save_path, 'pixels.csv')
    if os.path.exists(save_csv):
        if not args.force:
            print(f'File already exists, please delete it first or direct use the {os.path.abspath(save_csv)}, or use --force to overwrite it.')
            return 
        else:
            print(f'File already exists, but --force is used, so overwriting the {os.path.abspath(save_csv)}.')

    print('Please left click and drag to select points on the image. Close the window when done.')
    print('Note: This requires a screen display or X11 forwarding to work properly.')
    assert os.environ.get('DISPLAY') is not None, "DISPLAY environment variable is not set. Please ensure you have a screen display or X11 forwarding."

    sat_image = plt.imread(args.input_img_path)
    pixels, angles, smooth_path,fig_final = select_points(sat_image,args.num_of_point)
    fig_final.savefig(os.path.join(save_path,'trajectory.png'),dpi=300)

    with open(save_csv, 'w') as f:
        f.write('w,h,angle\n')
        for i in range(len(smooth_path)-1):
            f.write(f'{smooth_path[i][0]},{smooth_path[i][1]},{angles[i]}\n')
    print('the visualization of trajectory and visualize img is saved to:', os.path.abspath(save_csv))

if __name__ == '__main__':
    dragging = False
    main()
