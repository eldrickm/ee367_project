#! /usr/bin/env python3
"""
Simulation of out-of-focus blur and pre-conditioning for a projector-camera
system.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from fspecial import fspecial_gaussian_2d
import camera_tools.fullscreen.fullscreen as fs

# Key bindings to display different images on projector
QUIT_KEY = 'q'
ORIGINAL_KEY = 'a'
PRESHARP_KEY = 's'
DOT_KEY = 'd'
ITERATIVE_KEY = 'f'


def create_circle(radius: int, img_height: int, img_width: int):
    """
    Used to create a 2D filter with a given radius in pixels
    """
    x = np.arange(-img_height // 2, img_height // 2, 1)
    y = np.arange(-img_width // 2, img_width // 2, 1)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2 <= radius**2
    return z


def filter_rgb(img, otf):
    """
    Frequency domaing filtering across RGB color channels
    """
    filtered = np.zeros(img.shape)
    filtered[:, :, 0] = ifft2(fft2(img[:, :, 0]) * otf).real
    filtered[:, :, 1] = ifft2(fft2(img[:, :, 1]) * otf).real
    filtered[:, :, 2] = ifft2(fft2(img[:, :, 2]) * otf).real
    return filtered


def main(filepath='img/test.JPEG'):
    # Load original image
    original = io.imread(filepath) / 255

    # Simulated Forward Image Formation
    S_PSF_DIMENSIONS = (50, 50)
    S_PSF_SIGMA = 5

    s_psf = fspecial_gaussian_2d(S_PSF_DIMENSIONS, S_PSF_SIGMA)
    s_otf = psf2otf(s_psf, original.shape[:2])

    # Blur image
    sim_blur = filter_rgb(original, s_otf)

    # Plot Original Image / Simulated PSF / Simulated Blurred Image
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 30))
    ax[0].imshow(original, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(sim_blur, cmap='gray')
    ax[1].set_title('Simulated Blurred Image PSNR: %0.3f dB'
                    % (compute_psnr(original, sim_blur)))
    ax[1].axis('off')

    ax[2].imshow(s_psf, cmap='gray')
    ax[2].set_title('Simulated PSF (%d x %d, sigma = %0.2f)'
                    % (*S_PSF_DIMENSIONS, S_PSF_SIGMA))
    ax[2].axis('off')
    plt.savefig('fig/simulated.png', dpi=100, bbox_inches='tight')


    # Wiener Deconvolution for Preconditioning
    E_PSF_DIMENSIONS = (50, 50)
    E_PSF_SIGMA = 1
    SNR = np.mean(original) / 0.01

    e_psf = fspecial_gaussian_2d(E_PSF_DIMENSIONS, E_PSF_SIGMA)
    e_otf = psf2otf(e_psf, original.shape[:2])

    # Wiener Pre-Filtering
    wiener_filt = e_otf.conj() / (np.abs(e_otf ** 2) + 1 / SNR)
    precond = filter_rgb(original, wiener_filt)
    precond = np.clip(precond, 0, 1)

    # Simulate forward blur on preconditioned image
    wiener_result = filter_rgb(precond, s_otf)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 30))
    ax[0].imshow(precond, cmap='gray')
    ax[0].set_title('Preconditioned Image (Wiener Deconvolution)')
    ax[0].axis('off')

    ax[1].imshow(wiener_result, cmap='gray')
    ax[1].set_title('Simulated Result (Wiener Deconvolution) PSNR: %0.3f dB'
                    % (compute_psnr(original, wiener_result)))
    ax[1].axis('off')

    ax[2].imshow(e_psf, cmap='gray')
    ax[2].set_title('Estimated PSF (%d x %d, sigma = %0.2f)'
                    % (*E_PSF_DIMENSIONS, E_PSF_SIGMA))
    ax[2].axis('off')
    plt.savefig('fig/wiener.png', dpi=100, bbox_inches='tight')


    # # Iterative Pre-Filtering from Zhang and Nayar 2006
    # p = original
    # blurred = np.zeros(p.shape)
    # fg = np.zeros(p.shape)
    # for i in range(2):
    #     blurred = filter_rgb(p, e_otf)
    #     g = original - blurred
    #     g = filter_rgb(g, wiener_filt)
    #     fg = filter_rgb(g, e_otf)
    #     n = (np.linalg.norm(g) ** 2) / (np.linalg.norm(fg) ** 2)
    #     p_tilde = p + n * g
    #     p = np.clip(p_tilde, 0, 1)

    # # Simulate forward blur on preconditioned image
    # zhang_result = filter_rgb(p, s_otf)
    # zhang_result = np.clip(zhang_result, 0, 1)

    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 30))
    # ax[0].imshow(p, cmap='gray')
    # ax[0].set_title('Preconditioned Image (Zhang and Nayar 2006)')
    # ax[0].axis('off')

    # ax[1].imshow(zhang_result, cmap='gray')
    # ax[1].set_title('Simulated Result (Zhang and Nayar 2006) PSNR: %0.3f dB'
    #                 % (compute_psnr(original, zhang_result)))
    # ax[1].axis('off')

    # ax[2].imshow(e_psf, cmap='gray')
    # ax[2].set_title('Estimated PSF (%d x %d, sigma = %0.2f)'
    #                 % (*E_PSF_DIMENSIONS, E_PSF_SIGMA))
    # ax[2].axis('off')
    # plt.savefig('fig/zhang.png', dpi=100, bbox_inches='tight')


    # Plot Original Image / Simulated PSF / Simulated Blurred Image
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(30, 30))
    ax[0].imshow(original, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(precond, cmap='gray')
    ax[1].set_title('Preconditioned Image (%d x %d, sigma = %0.2f)'
                    % (*E_PSF_DIMENSIONS, E_PSF_SIGMA))
    ax[1].axis('off')

    ax[2].imshow(sim_blur, cmap='gray')
    ax[2].set_title('Simulated Blurred Image (%d x %d, sigma = %0.2f) PSNR: %0.3f dB'
                    % (*S_PSF_DIMENSIONS, S_PSF_SIGMA,
                       compute_psnr(original, sim_blur)))
    ax[2].axis('off')

    ax[3].imshow(wiener_result, cmap='gray')
    ax[3].set_title('Our Result PSNR: %0.3f dB'
                    % (compute_psnr(original, wiener_result)))
    ax[3].axis('off')


    plt.savefig('fig/result.png', dpi=100, bbox_inches='tight')


    # Uncomment below to test out projection
    # Single dot to project for PSF estimation
    # PROJECTOR_RESOLUTION = (1920, 1080)
    # DOT_RADIUS = 10                        # In Pixels
    # DISTANCE_FROM_FOCAL_PLANE = 1          #
    # dot_image = create_circle(DOT_RADIUS, *PROJECTOR_RESOLUTION).astype(np.uint8)
    # dot_image = dot_image * 255

    # Project Images
    # original = (original * 255).astype(np.uint8)
    # precond = (precond * 255).astype(np.uint8)
    # p = (p * 255).astype(np.uint8)

    # screen = fs.FullScreen(0)
    # screen.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    # while True:
    #     if cv2.waitKey(0) & 255 == ord(ORIGINAL_KEY):
    #         screen.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    #     if cv2.waitKey(0) & 255 == ord(PRESHARP_KEY):
    #         screen.imshow(cv2.cvtColor(precond, cv2.COLOR_BGR2RGB))
    #     if cv2.waitKey(0) & 255 == ord(DOT_KEY):
    #         screen.imshow(dot_image)
    #     if cv2.waitKey(0) & 255 == ord(ITERATIVE_KEY):
    #         screen.imshow(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
    #     if cv2.waitKey(0) & 255 == ord(QUIT_KEY):
    #         break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main('img/test.JPEG')
