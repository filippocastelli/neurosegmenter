from matplotlib import pylab as plt
from matplotlib.widgets import Slider, Button
from skimage import exposure
import numpy as np
import tensorflow as tf


def BatchInspector(config, batch, title: str = None):
    if config.training_mode == "2d":
        return BatchInspector2D(batch, title=title)
    elif config.training_mode == "3d":
        return BatchInspector3D(batch, title=title)
    else:
        raise NotImplementedError(config.training_mode)


class BatchInspectorBase:
    def __init__(self,
                 batch,
                 title=None,
                 keymap_conflicts={"k", "j"},
                 block=True
                 ):
        self.block = block
        frames_batch, masks_batch = batch
        if tf.is_tensor(frames_batch):
            self.frames_batch = frames_batch.numpy()
            self.masks_batch = masks_batch.numpy()
        else:
            self.frames_batch = frames_batch
            self.masks_batch = masks_batch
        self._fix_dims()
        self.fig, self.axs = plt.subplots(ncols=2)
        self.n_imgs = self.frames_batch.shape[0]
        self.frames_ax, self.masks_ax = self.axs
        self.title = title if title is not None else "BATCH INSPECTOR"
        self.keymap_conflicts = keymap_conflicts
        self._remove_keymap_conflicts(self.keymap_conflicts)
        self._create_btns()
        self.start_inspector()

    def start_inspector(self):
        self.index = 0
        self._init_figure()
        self._update_titles()
        self.fig.canvas.mpl_connect("key_press_event", self._process_key)
        plt.show(block=self.block)

    def _fix_dims(self):
        if self.frames_batch.shape[-1] == 2:
            zeros = np.zeros_like(self.frames_batch[..., 0])
            zeros = np.expand_dims(zeros, axis=-1)
            self.frames_batch = np.concatenate([self.frames_batch, zeros], axis=-1)

    def _init_figure(self):
        self._init_subplots()
        self._init_widgets()
        self.fig.suptitle(self.title, fontsize=16)

    def _init_subplots(self):
        self.first_subplot = self._init_subplot(self.frames_ax, self.frames_batch, set_clim=False)
        self.second_subplot = self._init_subplot(self.masks_ax, self.masks_batch)

    def _init_subplot(self, ax, array, set_clim=True):
        handler = ax.imshow(array[self.index])
        if set_clim:
            handler.set_clim(0, 1)
        return handler

    def _init_widgets(self):
        self.gamma_ax = self.fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor="white")
        self.gamma_slider = Slider(self.gamma_ax, 'Gamma', 0.1, 2, valinit=1.0, valstep=0.01)
        self.gamma_slider.on_changed(self._update_widgets)
        self.gamma_gt_ax = self.fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor="white")
        self.gamma_slider_gt = Slider(self.gamma_gt_ax, "Gamma GT", 0.1, 2, valinit=1.0, valstep=0.01)
        self.gamma_slider_gt.on_changed(self._update_widgets)

    def _update_widgets(self, val):
        current_img = self.frames_batch[self.index]
        current_gt = self.masks_batch[self.index]
        gamma = self.gamma_slider.val
        gamma_gt = self.gamma_slider_gt.val
        transformed_img = exposure.adjust_gamma(current_img, gamma=gamma)
        transformed_gt = exposure.adjust_gamma(current_gt, gamma=gamma_gt)
        self.frames_ax.images[0].set_array(transformed_img)
        self.masks_ax.images[0].set_array(transformed_gt)
        self.fig.canvas.draw_idle()

    def _reset_widgets(self, event):
        self.gamma_slider.reset()

    def _process_key(self, event):
        if event.key == 'j':
            self._next_slice()
        elif event.key == 'k':
            self._previous_slice()
        self.fig.canvas.draw()

    def _next_slice(self, event=None):
        new_idx = (self.index + 1) % self.n_imgs
        self._change_idx(new_idx)

    def _previous_slice(self, event=None):
        new_idx = (self.index - 1) % self.n_imgs
        self._change_idx(new_idx)

    def _change_idx(self, idx):
        self.index = idx
        # self.frames_ax.images[0].set_array(self.frames_batch[idx])
        self._update_widgets(self.gamma_slider.val)
        self.masks_ax.images[0].set_array(self.masks_batch[idx])
        self._update_titles()

    def _update_titles(self):
        for ax in self.axs:
            ax.set_title("Slice n {}".format(self.index))

    def _create_btns(self):
        self.axprev = plt.axes([0.7, 0.005, 0.1, 0.07])
        self.axnext = plt.axes([0.81, 0.005, 0.1, 0.07])

        self.bprev = Button(self.axprev, "<")
        self.bnext = Button(self.axnext, ">")

        self.bprev.on_clicked(self._previous_slice)
        self.bnext.on_clicked(self._next_slice)

    @staticmethod
    def _remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set

                for key in remove_list:
                    keys.remove(key)


class BatchInspector2D(BatchInspectorBase):
    def __init__(self,
                 batch,
                 title: str = None):
        super().__init__(batch, title)


class BatchInspector3D(BatchInspectorBase):
    def __init__(self, batch, title: str = None):
        self.title = title
        self.frame3d_idx = 0
        self.frames, self.masks = batch
        self.frame3d = self.frames[self.frame3d_idx]
        self.mask3d = self.masks[self.frame3d_idx]
        self.n_3d_frames = len(self.frames)
        keymap_conflicts = {"k", "j", "i", "u"}
        super().__init__(batch=(self.frame3d, self.mask3d),
                         keymap_conflicts=keymap_conflicts,
                         block=False,
                         title=self.title)
        self._create_btns3d()
        plt.show(block=True)

    def _process_key(self, event):
        if event.key == "j":
            self._next_slice()
        elif event.key == "k":
            self._previous_slice()
        elif event.key == "u":
            self._next_elem()
        elif event.key == "i":
            self._prev_elem()

    def _next_elem(self, event=None):
        self.frame3d_idx = (self.frame3d_idx + 1) % self.n_3d_frames
        self._set_3d_idx(self.frame3d_idx)

    def _prev_elem(self, event=None):
        self.frame3d_idx = (self.frame3d_idx - 1) % self.n_3d_frames
        self._set_3d_idx(self.frame3d_idx)

    def _set_3d_idx(self, idx):
        self.index = 0
        self.frames_batch = self.frames[idx]
        self.masks_batch = self.masks[idx]
        self.masks_ax.images[0].set_array(self.masks_batch[idx])
        self.fig.canvas.draw()
        self._update_titles()
        self._update_widgets(self.gamma_slider.val)

    def _create_btns3d(self):
        self.axprev_new = plt.axes([0.45, 0.005, 0.1, 0.07])
        self.axnext_new = plt.axes([0.56, 0.005, 0.1, 0.07])

        self.buttonprev_new = Button(self.axprev_new, "<<")
        self.buttonnext_new = Button(self.axnext_new, ">>")

        self.buttonprev_new.on_clicked(self._prev_elem)
        self.buttonnext_new.on_clicked(self._next_elem)

        # self.axnew = plt.axes([0,0,1,1])
        # self.button = Button(self.axnew, "asdasd")

        # self.button.on_clicked(self.new_fn)

    def new_fn(self, event=None):
        print("ciao")

    def _update_titles(self):
        for ax in self.axs:
            ax.set_title(
                "Sample {}/{}, Slice {}/{}".format(self.frame3d_idx, self.n_3d_frames, self.index, self.n_imgs))
