# HRPT Image Correction Tool

A simple Python tool for turning corrupted HRPT imagery into clean, beautiful images :D

I'm not a professional programmer. This project was developed with the help of AI coding tools, but it works, and I'm happy with the results.

## How to use

1. Decode your HRPT data in [SatDump](https://github.com/SatDump/SatDump) as usual.
2. Drag and drop the corrupted channel files into the program.
3. Select **HRPT** mode.
4. Adjust the parameters until all dark areas you want to correct are highlighted.
5. Run the correction. The script will replace images with the corrected ones. Images without correction will be stored in backup folder.
6. Load `product.cbor` into SatDump and process the data as usual.

## Feedback and contributions

Feedback, bug reports, and suggestions for improvement are welcome! I'd like to refine the tool and make it as reliable and useful as possible.

If you encounter any issues or have ideas for improving the correction algorithm, feel free to open an issue or share your thoughts.
