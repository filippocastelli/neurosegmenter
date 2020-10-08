from matplotlib import pylab as plt
from matplotlib.widgets import Slider, Button
from skimage import exposure
import numpy as np

class BatchInspector2D:
    def __init__(self,
                 batch):
        self.frames_batch, self.masks_batch = batch
        self._fix_dims()
        self.frames_original = self.frames_batch
        self.fig, self.axs = plt.subplots(ncols=2)
        self.n_imgs = self.frames_batch.shape[0]
        self.frames_ax, self.masks_ax = self.axs
        self._remove_keymap_conflicts({"k", "j"})
        self.start_inspector()
    
    def start_inspector(self):
        self.index=0
        self._init_figure()
        self._update_titles()
        self.fig.canvas.mpl_connect("key_press_event", self._process_key)
        plt.show(block=True)
        
    def _fix_dims(self):
        if self.frames_batch.shape[-1] == 2:
            zeros = np.zeros_like(self.frames_batch[...,0])
            zeros = np.expand_dims(zeros, axis=-1)
            self.frames_batch = np.concatenate([self.frames_batch, zeros], axis=-1)
        
    def _init_figure(self):
        self._init_subplots()
        self._init_widgets()
        self.fig.suptitle("BATCH INSPECTOR v0.01", fontsize=16)
        
    def _init_widgets(self):
        self.gamma_ax = self.fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor="white")
        self.gamma_slider = Slider(self.gamma_ax, 'Gamma', 0.1, 2, valinit=1.0, valstep=0.01)
        self.gamma_slider.on_changed(self._update_widgets)
        
    def _update_widgets(self, val):
        current_img = self.frames_batch[self.index]
        gamma = self.gamma_slider.val
        transformed_img = exposure.adjust_gamma(current_img, gamma=gamma)
        self.frames_ax.images[0].set_array(transformed_img)
        self.fig.canvas.draw_idle()
    
    def _reset_widgets(self,event):
        self.gamma_slider.reset()
        
    def _init_subplot(self, ax, array, set_clim=True):
        handler = ax.imshow(array[self.index])
        if set_clim:
            handler.set_clim(0,1)
        return handler
    
    def _init_subplots(self):
        self.first_subplot = self._init_subplot(self.frames_ax, self.frames_batch, set_clim=False)
        self.second_subplot = self._init_subplot(self.masks_ax,self.masks_batch)
        
    def _process_key(self, event):
        if event.key == 'j':
            self._next_slice()
        elif event.key == 'k':
            self._previous_slice()
        self.fig.canvas.draw()
     
    def _next_slice(self):
        new_idx = (self.index + 1) % self.n_imgs
        self._change_idx(new_idx)
        
    def _previous_slice(self):
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
            
    @staticmethod
    def _remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                
                for key in remove_list:
                    keys.remove(key)
    
        
        