import py4DSTEM
from PyQt5.QtWidgets import QFileDialog
import h5py


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

def load_data_EMPAD2(self,):
    import numpy as np
    # Load EMPAD2 data and perform combination of concatenated binary data
    filepath = self.show_file_dialog()

    # get file size
    import os
    filesize = os.path.getsize(filepath)
    pattern_size = 128 * 128 * 4  # 4 bytes per pixel
    N_patterns = filesize / pattern_size
    Nxy = np.sqrt(N_patterns)

    # Check that it's reasonably square
    assert (
        np.abs(Nxy - np.round(Nxy)) <= 1e-10
    ), "Automatically detected shape seems wrong... Try specifying it manually with the EMPAD_shape keyword argument"

    data_shape = (int(Nxy), int(Nxy), 128, 128)

    with open(filepath, "rb") as fid:
        self.datacube = py4DSTEM.DataCube(
            np.fromfile(fid, np.uint32).reshape(data_shape)
        )

    # open the calibration files
    cal_names = ["G1A","G2A","G1B","G2B","FFA","FFB","B2A","B2B"]
    G1A, G2A, G1B, G2B, FFA, FFB, B2A, B2B = [
        np.fromfile(
            os.path.join(
                os.path.dirname(__file__), f"EMPAD2_calibrations/{calname}.r32",
                ),
            dtype=np.float32, count=128*128
        ).reshape(128,128) for calname in cal_names
    ]

    # apply calibration to each pattern (this might be better batched differently, such as per row!)
    for rx,ry in py4DSTEM.tqdmnd(data_shape[0], data_shape[1]):
        data = self.datacube.data[rx,ry]
        analog = np.bitwise_and(data, 0x3fff).astype(np.float32)
        digital = np.bitwise_and(data, 0x3fffc000).astype(np.float32)
        gain = np.bitwise_and(data, 0x80000000) != 0

        # currently this does not do debouncing!
        if ry % 2:
            # odd frame
            self.datacube.data[rx,ry] = FFB * (analog * (1-gain) + G1B * (analog - B2B) * gain + G2A * digital)
        else:
            # even frame
            self.datacube.data[rx,ry] = FFA * (analog * (1-gain) + G1A * (analog - B2A) * gain + G2A * digital)

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filepath)

def load_file(self, filepath, mmap=False, binning=1):
    print(f"Loading file {filepath}")

    if py4DSTEM.io.utils.parse_filetype(filepath) == "py4DSTEM":
        datacubes = get_4D(h5py.File(filepath, "r"))
        print(f"Found {len(datacubes)} 4D datasets inside the HDF5 file...")
        if len(datacubes) >= 1:
            # Read the first datacube in the HDF5 file into RAM
            print(f"Reading dataset at location {datacubes[0].name}")
            self.datacube = py4DSTEM.io.DataCube(
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
