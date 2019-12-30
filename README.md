# CVAEs for Anomaly Detection to Reduce Bandwidth in Streaming Video

Modern compression techniques leverage a learned representation over an image distribution to more efficiently encode redundant spatial context for a sample. Video content is also generally redundant in time. 

This work implements online training of convolutional autoencoders to learn a nonstationary image distribution in streaming video. Then, we perform anomaly detection by thresholding the autoencoder's reconstruction loss, which we regard as an anomaly score.

![anomaly_detected](cvae_anomaly_detection.gif)

## Setup

Adapt the included config.ini to suit your needs. 

The demo uses OpenCV so you can configure the video sources with:
 * a webcam index  (0,1 or /dev/video0)
 * an RTSP url 
 * other URIs recognized by cv2.VideoCapture() (.mp4 or .mkv containers)

To affect model recency bias, consider changing:
 * the model learning rate (smaller rates --> less recency bias)
 * the time window length (smaller windows --> greater recency bias)

To explore different network architectures, change the net_arch parameter. This should be a list of 3-tuples organized as:
 * First entry - No. of Filters (Output dim)
 * Second entry - Size of Convolutional Kernel (assumed square)
 * Third entry - Size of Pooling (assumed square)

## Dependencies

This repo requires opencv and tensorflow:
```
pip install -r requirements.txt
```

## Running the Demo

Set the config to reflect your video sources and model preferences before running:

```
python3 visual_anomaly_detection_demo.py
```

A CVAE is instantiated for each video source according to preferences set in the config.ini. 

Online training begins and normal/anomaly classes along with the reconstruction loss will be streamed to stdout according to the configured threshold (by default 5 sigma deviation from moving average).

## References

 * [Hackster/NVIDIA](https://www.hackster.io/contests/NVIDIA) - AI at the Edge Challenge [Entry](https://www.hackster.io/smellslikeml/saving-bandwidth-with-anomaly-detection-16eb67)
