**Requirement**: *video shall have a resolution of **480p** and a framerate of **30 fps**.*

| Horizontal | Vertical | Color Channels  | Color Depth | Framerate |
|:----------:|:--------:|:---------------:|:-----------:|:---------:|
| 640 px     | 480 px   | 3-channel (RGB) | 8-bit       | 30 fps    |

So, each frame being inferenced by either encoder or decoder has an input 
640 * 480 * 3 (*8) * 30 = 221,184,000 bps (27,648,000 Bps)

$$out(N_i, C_{{out}_j}) = bias(C_{{out}_j}) + \sum_{k=0}^{C_{in}-1}​ weight(C_{{out}_j}, k)⋆input(N_i, k)$$

> "Vector multiplication instructions are designed to perform the multiplication and access 16 bytes at the same time" [[1]](#1) (1.6.4).

That is, 

## References
<a id="1">[1]</a>
[ESP32 Manual](https://www.espressif.com/sites/default/files/documentation/esp32-s3_technical_reference_manual_en.pdf)