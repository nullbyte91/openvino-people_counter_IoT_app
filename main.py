"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

from fysom import *
from imutils.video import FPS

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Browser and OpenCV Window toggle
Browser_ON = False

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

fsm = Fysom({'initial': 'empty',
             'events': [
                 {'name': 'enter', 'src': 'empty', 'dst': 'standing'},
                 {'name': 'exit',  'src': 'standing',   'dst': 'empty'}]})

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-o", "--output", type=str, default="LOCAL",
                        help="Output window local or Web Server (use -o WEB)"
                        "(LOCAL by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client
    client =  mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialize the Inference Engine
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device, CPU_EXTENSION, num_requests=0)

    # Get a Input blob shape
    in_n, in_c, in_h, in_w = infer_network.get_input_shape()

    # Get a output blob name
    _ = infer_network.get_output_name()
    
    # Handle the input stream
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    _, frame = cap.read()

    people_total_count = 0
    people_in_a_frame = 0

    g_elapsed = 0
    entre_ROI_xmin = 400
    entre_ROI_ymin = 450
    exit_ROI_xmin = 550
    exit_ROI_ymin = 410

    fps = FPS().start()

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        
        fh = frame.shape[0]
        fw = frame.shape[1]
        key_pressed = cv2.waitKey(50)

        # Pre-process the frame
        image_resize = cv2.resize(frame, (in_w, in_h))
        image = image_resize.transpose((2,0,1))
        image = image.reshape(in_n, in_c, in_h, in_w)
        
        # Perform inference on the frame
        infer_network.exec_net(image, request_id=0)
        
        # Get the output of inference
        if infer_network.wait(request_id=0) == 0:
            result = infer_network.get_output(request_id=0)
            for box in result[0][0]: # Output shape is 1x1x100x7
                conf = box[2]
                if conf >= prob_threshold:
                    xmin = int(box[3] * fw)
                    ymin = int(box[4] * fh)
                    xmax = int(box[5] * fw)
                    ymax = int(box[6] * fh)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

                    if xmin < entre_ROI_xmin and ymax < entre_ROI_ymin:  
                            if fsm.current == "empty":
                                # Count a people
                                people_in_a_frame += 1
                                people_total_count += 1
                                # Start the timer
                                start_time = time.perf_counter()
                                # Person entered a room - fsm state change
                                fsm.enter()
                                print(xmax, ymax)
                                if args.output == "WEB":
                                    # Publish people_count messages to the MQTT server
                                    client.publish("person", json.dumps({"count": people_in_a_frame}))
                                log.info("#########################")
                                log.info("Person entered into frame")
                                log.info("#########################")

                    if xmin > exit_ROI_xmin and ymax < exit_ROI_ymin:
                        if fsm.current == "standing":
                            # Change the state to exit - fsm state change
                            fsm.exit()
                            stop_time = time.perf_counter()
                            elapsed = stop_time - start_time
                            
                            # Update average time
                            log.info("elapsed time = {:.12f} seconds".format(elapsed))
                            g_elapsed = (g_elapsed + elapsed) / people_total_count
                            log.info("g_elapsed time = {:.12f} seconds".format(g_elapsed))
                            
                            people_in_a_frame = 0

                            if args.output == "WEB":
                                # Publish duration messages to the MQTT server
                                client.publish("person/duration", json.dumps({"duration": g_elapsed}))
                                client.publish("person", json.dumps({"count": people_in_a_frame}))
                            log.info("#########################")
                            log.info("Person exited from frame")
                            log.info("#########################")

                    log.info("xmin:{} xmax:{} ymin:{} ymax:{}".format(xmin, xmax, ymin, ymax))
                
        if args.output != "WEB":                                                
            # Update info on frame
            info = [
                ("people_ccount", people_total_count),
            ]
            
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, fh - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if args.output == "WEB":
            # Push to FFmpeg server
            sys.stdout.buffer.write(frame)

            sys.stdout.flush()
        else:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #Break if escape key pressed
            if key_pressed == 27:
                break
        
        fps.update()
    
    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()

    if args.output == "WEB":
        client.disconnect()
    else:
        cv2.destroyAllWindows()
    
    fps.stop()

    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Set log to INFO
    log.basicConfig(level=log.INFO)

    # Grab command line args
    args = build_argparser().parse_args()
    if args.output == "WEB":
        # Connect to the MQTT server
        client = connect_mqtt()
    else:
        client = None 

    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()