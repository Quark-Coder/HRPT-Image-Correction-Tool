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

<img width="2636" height="701" alt="not_corrected" src="https://github.com/user-attachments/assets/cb6a1fa4-c934-4c83-a96b-e7b9f43e805e" />
<img width="2636" height="701" alt="corrected" src="https://github.com/user-attachments/assets/0501dc72-7943-4e03-96b4-6a51a2947bfb" />
<img width="2800" height="3895" alt="meteor_not_corrected" src="https://github.com/user-attachments/assets/49e6cca4-fcff-437d-a9f3-313271d862aa" />
<img width="2800" height="3895" alt="meteor_corrected" src="https://github.com/user-attachments/assets/72dd0b68-0b47-4c1c-837d-bd1b1e8ce5c6" />


