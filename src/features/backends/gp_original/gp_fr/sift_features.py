import numpy as np
from scipy import signal

Nangles = 8
Nbins = 4
Nsamples = Nbins ** 2
alpha = 9.0
angles = np.array(range(Nangles)) * 2.0 * np.pi / Nangles


def gen_dgauss(sigma):
    fwid = int(2 * np.ceil(sigma))
    G = np.array(range(-fwid, fwid + 1)) ** 2
    G = G.reshape((G.size, 1)) + G
    G = np.exp(-G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH, GW = np.gradient(G)
    GH *= 2.0 / np.sum(np.abs(GH))
    GW *= 2.0 / np.sum(np.abs(GW))
    return GH, GW


class DsiftExtractor:
    def __init__(self, gridSpacing, patchSize, nrml_thres=1.0,
                 sigma_edge=0.8, sift_thres=0.2):
        self.gS = gridSpacing
        self.pS = patchSize
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        sample_res = self.pS / np.double(Nbins)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p, sample_p)
        sample_ph = sample_ph.flatten()
        sample_pw = sample_pw.flatten()
        bincenter = np.array(range(1, Nbins * 2, 2)) / 2.0 / Nbins * self.pS - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter, bincenter)
        bincenter_h = bincenter_h.reshape((bincenter_h.size, 1))
        bincenter_w = bincenter_w.reshape((bincenter_w.size, 1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1 - weights_h) * (weights_h <= 1)
        weights_w = (1 - weights_w) * (weights_w <= 1)
        self.weights = weights_h * weights_w

    def process_image(self, image, positionNormalize=True, verbose=False):
        image = image.astype(np.double)
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        H, W = image.shape
        gS = self.gS
        pS = self.pS
        remH = np.mod(H - pS, gS)
        remW = np.mod(W - pS, gS)
        offsetH = int(remH / 2)
        offsetW = int(remW / 2)
        gridH, gridW = np.meshgrid(
            range(offsetH, H - pS + 1, gS),
            range(offsetW, W - pS + 1, gS),
        )
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        feaArr = self.calculate_sift_grid(image, gridH, gridW)
        feaArr = self.normalize_sift(feaArr)
        if positionNormalize:
            positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        else:
            positions = np.vstack((gridH, gridW))
        return feaArr, positions

    def calculate_sift_grid(self, image, gridH, gridW):
        H, W = image.shape
        Npatches = gridH.size
        feaArr = np.zeros((Npatches, Nsamples * Nangles))
        GH, GW = gen_dgauss(self.sigma)
        IH = signal.convolve2d(image, GH, mode='same')
        IW = signal.convolve2d(image, GW, mode='same')
        Imag = np.sqrt(IH ** 2 + IW ** 2)
        Itheta = np.arctan2(IH, IW)
        Iorient = np.zeros((Nangles, H, W))
        for i in range(Nangles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - angles[i]) ** alpha, 0)
        for i in range(Npatches):
            currFeature = np.zeros((Nangles, Nsamples))
            for j in range(Nangles):
                currFeature[j] = np.dot(
                    self.weights,
                    Iorient[j, gridH[i]:gridH[i] + self.pS,
                            gridW[i]:gridW[i] + self.pS].flatten(),
                )
            feaArr[i] = currFeature.flatten()
        return feaArr

    def normalize_sift(self, feaArr):
        siftlen = np.sqrt(np.sum(feaArr ** 2, axis=1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        feaArr /= siftlen.reshape((siftlen.size, 1))
        feaArr[feaArr > self.sift_thres] = self.sift_thres
        if np.any(hcontrast):
            feaArr[hcontrast] /= np.sqrt(
                np.sum(feaArr[hcontrast] ** 2, axis=1)
            ).reshape((feaArr[hcontrast].shape[0], 1))
        return feaArr


class SingleSiftExtractor(DsiftExtractor):
    def __init__(self, patchSize, nrml_thres=1.0, sigma_edge=0.8, sift_thres=0.2):
        super().__init__(patchSize, patchSize, nrml_thres, sigma_edge, sift_thres)

    def process_image(self, image):
        return DsiftExtractor.process_image(self, image, False, False)[0]
