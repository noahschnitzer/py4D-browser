import py4DSTEM
from PyQt5.QtWidgets import QFileDialog
import h5py
import numpy as np
import os

# Load EMPAD2 calibration files
cal_names = ["G1A", "G2A", "G1B", "G2B", "FFA", "FFB", "B2A", "B2B"]
_G1A, _G2A, _G1B, _G2B, _FFA, _FFB, _B2A, _B2B = [
    np.fromfile(
        os.path.join(
            os.path.dirname(__file__),
            f"EMPAD2_calibrations/{calname}.r32",
        ),
        dtype=np.float32,
        count=128 * 128,
    ).reshape(128, 128)
    for calname in cal_names
]


def load_data_auto(self):
    filename = self.show_file_dialog()
    self.load_file(filename)


def load_data_mmap(self):
    filename = self.show_file_dialog()
    self.load_file(filename, mmap=True)


def load_data_bin(self):
    # TODO: Ask user for binning level
    filename = self.show_file_dialog()
    self.load_file(filename, mmap=False, binning=4)


def load_data_EMPAD2(
    self,
):
    # Load EMPAD2 data and perform combination of concatenated binary data
    filepath = self.show_file_dialog()

    if hasattr(self, "_EMPAD_bkg_odd"):
        self.datacube = _load_EMPAD2_datacube(
            filepath, self._EMPAD_bkg_even, self._EMPAD_bkg_odd
        )
        self.setWindowTitle(filepath + " (Background Subtracted)")
    else:
        print("Warning! Loading without background correction!")
        self.datacube = _load_EMPAD2_datacube(filepath)
        self.setWindowTitle(filepath)

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)


def load_data_background_EMPAD2(
    self,
):
    filepath = self.show_file_dialog()

    print("Reading background data...")

    bkg_data = _load_EMPAD2_datacube(filepath)
    self._EMPAD_bkg_even = np.mean(bkg_data.data[:, ::2], axis=(0, 1))
    self._EMPAD_bkg_odd = np.mean(bkg_data.data[:, 1::2], axis=(0, 1))

    print(
        "Successfully loaded EMPAD2 background data... Ready to load experimental data."
    )


def load_file(self, filepath, mmap=False, binning=1):
    print(f"Loading file {filepath}")

    from py4DSTEM.io.parsefiletype import _parse_filetype

    if _parse_filetype(filepath) in ("H5", "legacy", "emd"):
        datacubes = get_4D(h5py.File(filepath, "r"))
        print(f"Found {len(datacubes)} 4D datasets inside the HDF5 file...")
        if len(datacubes) >= 1:
            # Read the first datacube in the HDF5 file into RAM
            print(f"Reading dataset at location {datacubes[0].name}")
            self.datacube = py4DSTEM.DataCube(
                datacubes[0] if mmap else datacubes[0][()]
            )
    else:
        self.datacube = py4DSTEM.import_file(
            filepath,
            mem="MEMMAP" if mmap else "RAM",
            binfactor=binning,
        )

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filepath)


def show_file_dialog(self):
    filename = QFileDialog.getOpenFileName(
        self,
        "Open 4D-STEM Data",
        "",
        "4D-STEM Data (*.dm3 *.dm4 *.raw *.mib *.gtg);;Any file (*)",
    )
    if filename is not None and len(filename[0]) > 0:
        return filename[0]
    else:
        print("File was invalid, or something?")
        raise ValueError("Could not read file")


def get_4D(f, datacubes=None):
    if datacubes is None:
        datacubes = []
    for k in f.keys():
        if isinstance(f[k], h5py.Dataset):
            # we found data
            if len(f[k].shape) == 4:
                datacubes.append(f[k])
        elif isinstance(f[k], h5py.Group):
            get_4D(f[k], datacubes)
    return datacubes


def _load_EMPAD2_datacube(filepath, background_even=None, background_odd=None):
    # get file size
    filesize = os.path.getsize(filepath)
    pattern_size = 128 * 128 * 4  # 4 bytes per pixel
    N_patterns = filesize / pattern_size
    Nxy = np.sqrt(N_patterns)

    # Check that it's reasonably square
    assert np.abs(Nxy - np.round(Nxy)) <= 1e-10, "Did you do a non-square scan?"

    data_shape = (int(Nxy), int(Nxy), 128, 128)

    with open(filepath, "rb") as fid:
        datacube = py4DSTEM.DataCube(np.fromfile(fid, np.float32).reshape(data_shape))

    _process_EMPAD2_datacube(datacube, background_even, background_odd)

    return datacube


def _process_EMPAD2_datacube(datacube, background_even=None, background_odd=None):
    # apply calibration to each pattern (this might be better batched differently, such as per row!)
    for rx, ry in py4DSTEM.tqdmnd(datacube.data.shape[0], datacube.data.shape[1]):
        data = datacube.data[rx, ry].view(np.uint32)
        analog = np.bitwise_and(data, 0x3FFF).astype(np.float32)
        digital = np.right_shift(np.bitwise_and(data, 0x3FFFC000), 14).astype(
            np.float32
        )
        gain_bit = np.right_shift(np.bitwise_and(data, 0x80000000), 31)

        # currently this does not do debouncing!
        if background_even is not None and background_odd is not None:
            if ry % 2:
                # odd frame
                datacube.data[rx, ry] = _FFB * (
                    (
                        analog * (1 - gain_bit)
                        + _G1B * (analog - _B2B) * gain_bit
                        + _G2B * digital
                    )
                    - background_odd
                )
            else:
                # even frame
                datacube.data[rx, ry] = _FFA * (
                    (
                        analog * (1 - gain_bit)
                        + _G1A * (analog - _B2A) * gain_bit
                        + _G2A * digital
                    )
                    - background_even
                )
        else:
            if ry % 2:
                # odd frame
                datacube.data[rx, ry] = (
                    analog * (1 - gain_bit)
                    + _G1B * (analog - _B2B) * gain_bit
                    + _G2B * digital
                )
            else:
                # even frame
                datacube.data[rx, ry] = (
                    analog * (1 - gain_bit)
                    + _G1A * (analog - _B2A) * gain_bit
                    + _G2A * digital
                )
