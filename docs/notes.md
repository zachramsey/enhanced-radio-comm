## Overview

**x** ->  
conv 128 5x5  
downsample 2x2  
GDN  
conv 128 5x5  
downsample 2x2  
GDN  
conv 128 5x5  
downsample 2x2  
GDN  
conv 192 5x5  
downsample 2x2  
-> **y**

**y** ->  
quantize  
-> **y_**

**y** ->  
ABS  
conv 128 3x3  
ReLU  
conv 128 5x5  
downsample 2x2  
ReLU  
conv 128 5x5  
downsample 2x2  
-> **z**

**z** ->  
quantize  
-> **z_**

**z_** ->  
arithmetic encoder  
-> **z_enc**

**transmit z_enc**

**z_enc** ->  
arithmetic decoder  
-> **z_**

**z_** ->  
conv 128 5x5  
upsample 2x2  
ReLU  
conv 128 5x5  
upsample 2x2  
ReLU  
conv 192 3x3  
ReLU  
-> **sig_**

**y_**, **sig_** ->  
arithmetic encoder  
-> **y_enc**

**transmit y_enc**

**y_enc**, **sig_** ->  
arithmetic decoder  
-> **y_**

**y_** ->  
conv 128 5x5  
upscale 2x2  
IGDN  
conv 128 5x5  
upscale 2x2  
IGDN  
conv 128 5x5  
upscale 2x2  
IGDN  
conv 3 5x5  
upscale 2x2  
-> **x_**

---

## Encoder

**x** ->  
conv 128 5x5  
downsample 2x2  
GDN  
conv 128 5x5  
downsample 2x2  
GDN  
conv 128 5x5  
downsample 2x2  
GDN  
conv 192 5x5  
downsample 2x2  
-> **y**

**y** ->  
ABS  
conv 128 3x3  
ReLU  
conv 128 5x5  
downsample 2x2  
ReLU  
conv 128 5x5  
downsample 2x2  
-> **z**

**z** ->  
quantize  
-> **z_**

**z_** ->  
arithmetic encoder  
-> **z_enc**

**transmit z_enc**

**z_** ->  
conv 128 5x5  
upsample 2x2  
ReLU  
conv 128 5x5  
upsample 2x2  
ReLU  
conv 192 3x3  
ReLU  
-> **sig_**

**y** ->  
quantize  
-> **y_**

**y_**, **sig_** ->  
arithmetic encoder  
-> **y_enc**

**transmit y_enc**

**y_enc**, **sig_** ->  
arithmetic decoder  
-> **y_**

---

## Decoder

**receive z_enc**

**z_enc** ->  
arithmetic decoder  
-> **z_**

**z_** ->  
conv 128 5x5  
upsample 2x2  
ReLU  
conv 128 5x5  
upsample 2x2  
ReLU  
conv 192 3x3  
ReLU  
-> **sig_**

**receive y_enc**

**y_enc**, **sig_** ->  
arithmetic decoder  
-> **y_**

**y_** ->  
conv 128 5x5  
upscale 2x2  
IGDN  
conv 128 5x5  
upscale 2x2  
IGDN  
conv 128 5x5  
upscale 2x2  
IGDN  
conv 3 5x5  
upscale 2x2  
-> **x_**
